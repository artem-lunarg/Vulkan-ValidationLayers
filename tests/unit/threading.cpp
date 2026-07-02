/*
 * Copyright (c) 2015-2026 The Khronos Group Inc.
 * Copyright (c) 2015-2026 Valve Corporation
 * Copyright (c) 2015-2026 LunarG, Inc.
 * Copyright (c) 2015-2026 Google, Inc.
 * Modifications Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <thread>
#include "layer_validation_tests.h"
#include "descriptor_helper.h"
#include "thread_helper.h"

#if GTEST_IS_THREADSAFE
class NegativeThreading : public VkLayerTest {};

TEST_F(NegativeThreading, CommandBufferCollision) {
    m_errorMonitor->SetDesiredError("THREADING ERROR");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");  // Ignore any extra threading errors found beyond the first one

    RETURN_IF_SKIP(Init());
    InitRenderTarget();

    // Test takes magnitude of time longer for profiles and slows down testing
    if (IsPlatformMockICD()) {
        GTEST_SKIP() << "Test not supported by MockICD";
    }

    // Calls AllocateCommandBuffers
    vkt::CommandBuffer commandBuffer(*m_device, m_command_pool);

    commandBuffer.Begin();

    vkt::Event event(*m_device);
    VkResult err;

    err = vk::ResetEvent(device(), event);
    ASSERT_EQ(VK_SUCCESS, err);

    ThreadTestData data;
    data.commandBuffer = commandBuffer;
    data.event = event;
    std::atomic<bool> bailout{false};
    data.bailout = &bailout;
    m_errorMonitor->SetBailout(data.bailout);

    // First do some correct operations using multiple threads.
    // Add many entries to command buffer from another thread.
    std::thread thread1(AddToCommandBuffer, &data);
    // Make non-conflicting calls from this thread at the same time.
    for (int i = 0; i < 1000 /* Initially 80000 to make machine miserable */; i++) {
        uint32_t count;
        vk::EnumeratePhysicalDevices(instance(), &count, NULL);
    }
    thread1.join();

    // Then do some incorrect operations using multiple threads.
    // Add many entries to command buffer from another thread.
    std::thread thread2(AddToCommandBuffer, &data);
    // Add many entries to command buffer from this thread at the same time.
    AddToCommandBuffer(&data);

    thread2.join();
    commandBuffer.End();

    m_errorMonitor->SetBailout(NULL);

    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeThreading, UpdateDescriptorCollision) {
    TEST_DESCRIPTION("Two threads updating the same descriptor set, expected to generate a threading error");

    m_errorMonitor->SetDesiredError("vkUpdateDescriptorSets(): THREADING ERROR");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");  // Ignore any extra threading errors found beyond the first one

    RETURN_IF_SKIP(Init());
    InitRenderTarget();

    OneOffDescriptorSet normal_descriptor_set(m_device,
                                              {
                                                  {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                              },
                                              0);

    vkt::Buffer buffer(*m_device, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    ThreadTestData data;
    data.device = device();
    data.descriptorSet = normal_descriptor_set.set_;
    data.binding = 0;
    data.buffer = buffer;
    std::atomic<bool> bailout{false};
    data.bailout = &bailout;
    m_errorMonitor->SetBailout(data.bailout);

    // Update descriptors from another thread.
    std::thread thread(UpdateDescriptor, &data);
    // Update descriptors from this thread at the same time.

    ThreadTestData data2;
    data2.device = device();
    data2.descriptorSet = normal_descriptor_set.set_;
    data2.binding = 1;
    data2.buffer = buffer;
    data2.bailout = &bailout;

    UpdateDescriptor(&data2);

    thread.join();

    m_errorMonitor->SetBailout(NULL);

    m_errorMonitor->VerifyFound();
}

// The following tests check that the threading error message reports the correct function pair:
// the current function and the function recorded for the other side of the race.
//
// The tests are constructed so that the set of valid message pairs is exact:
//  * each racing thread uses exactly one distinct function on the shared object, so the "other
//    thread" function in the message can only be the other thread's designated function;
//  * the shared object is warmed up (used once) by one of the racing threads before the race,
//    so even a stale read of the tracker state yields one of the designated functions and the
//    "and another thread" fallback cannot appear.
// The undesired regex (last SetDesiredFailureMsgRegex argument) fails the test if a message
// pairs functions that cannot conflict in the given scenario, i.e. if the message lies.

TEST_F(NegativeThreading, ReportFunctionPairSameFunction) {
    TEST_DESCRIPTION("Race vkSetEvent against vkSetEvent; both sides of the message must report vkSetEvent");

    m_errorMonitor->SetDesiredFailureMsgRegex(
        kErrorBit, "UNASSIGNED-Threading-MultipleThreads-Write",
        "vkSetEvent\\(\\): THREADING ERROR : object of type VkEvent is simultaneously used in current thread \\d+ "
        "\\(vkSetEvent\\) and thread \\d+ \\(vkSetEvent\\)",
        "and thread \\d+ \\((?!vkSetEvent\\))|and another thread|MultipleThreads-Read");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");  // Ignore any extra threading errors found beyond the first one

    RETURN_IF_SKIP(Init());

    vkt::Event event(*m_device);

    std::atomic<bool> bailout{false};
    m_errorMonitor->SetBailout(&bailout);

    vk::SetEvent(device(), event);  // warm up the tracker state from a racing thread

    const auto set_event_loop = [&]() {
        for (int i = 0; i < 80000 && !bailout; i++) {
            vk::SetEvent(device(), event);
        }
    };
    std::thread thread(set_event_loop);
    set_event_loop();
    thread.join();

    m_errorMonitor->SetBailout(NULL);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeThreading, ReportFunctionPairWriteWrite) {
    TEST_DESCRIPTION("Race vkSetEvent against vkResetEvent; the message must pair exactly these two functions");

    m_errorMonitor->SetDesiredFailureMsgRegex(
        kErrorBit, "UNASSIGNED-Threading-MultipleThreads-Write",
        "object of type VkEvent is simultaneously used in current thread \\d+ \\(vkSetEvent\\) and thread \\d+ "
        "\\(vkResetEvent\\)|"
        "object of type VkEvent is simultaneously used in current thread \\d+ \\(vkResetEvent\\) and thread \\d+ "
        "\\(vkSetEvent\\)",
        "\\(vkSetEvent\\) and thread \\d+ \\(vkSetEvent\\)|\\(vkResetEvent\\) and thread \\d+ \\(vkResetEvent\\)|"
        "and another thread|MultipleThreads-Read");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");  // Ignore any extra threading errors found beyond the first one

    RETURN_IF_SKIP(Init());

    vkt::Event event(*m_device);

    std::atomic<bool> bailout{false};
    m_errorMonitor->SetBailout(&bailout);

    vk::SetEvent(device(), event);  // warm up the tracker state from a racing thread

    std::thread thread([&]() {
        for (int i = 0; i < 80000 && !bailout; i++) {
            vk::ResetEvent(device(), event);
        }
    });
    for (int i = 0; i < 80000 && !bailout; i++) {
        vk::SetEvent(device(), event);
    }
    thread.join();

    m_errorMonitor->SetBailout(NULL);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeThreading, ReportFunctionPairReadWrite) {
    TEST_DESCRIPTION("Race vkSetEvent (write) against vkGetEventStatus (read); the message must pair exactly these two functions");

    // Both sides can report: the writer against the reader that owns the use window (Write VUID)
    // or the reader against the writer that owns it (Read VUID). Orientation of the pair follows the VUID.
    m_errorMonitor->SetDesiredFailureMsgRegex(
        kErrorBit, "",
        "MultipleThreads-Write[\\s\\S]*current thread \\d+ \\(vkSetEvent\\) and thread \\d+ \\(vkGetEventStatus\\)|"
        "MultipleThreads-Read[\\s\\S]*current thread \\d+ \\(vkGetEventStatus\\) and thread \\d+ \\(vkSetEvent\\)",
        "\\(vkSetEvent\\) and thread \\d+ \\(vkSetEvent\\)|\\(vkGetEventStatus\\) and thread \\d+ \\(vkGetEventStatus\\)|"
        "and another thread");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");  // Ignore any extra threading errors found beyond the first one

    RETURN_IF_SKIP(Init());

    vkt::Event event(*m_device);

    std::atomic<bool> bailout{false};
    m_errorMonitor->SetBailout(&bailout);

    vk::SetEvent(device(), event);  // warm up the tracker state from a racing thread

    std::thread thread([&]() {
        for (int i = 0; i < 80000 && !bailout; i++) {
            vk::GetEventStatus(device(), event);
        }
    });
    for (int i = 0; i < 80000 && !bailout; i++) {
        vk::SetEvent(device(), event);
    }
    thread.join();

    m_errorMonitor->SetBailout(NULL);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeThreading, ReportFunctionPairUnrelatedCalls) {
    TEST_DESCRIPTION(
        "Race vkSetEvent against vkResetEvent on a shared event while both threads also call other functions on their own "
        "(non-shared) objects. The message must never mention the unrelated functions");

    m_errorMonitor->SetDesiredFailureMsgRegex(
        kErrorBit, "UNASSIGNED-Threading-MultipleThreads-Write",
        "object of type VkEvent is simultaneously used in current thread \\d+ \\(vkSetEvent\\) and thread \\d+ "
        "\\(vkResetEvent\\)|"
        "object of type VkEvent is simultaneously used in current thread \\d+ \\(vkResetEvent\\) and thread \\d+ "
        "\\(vkSetEvent\\)",
        // Noise functions only touch per-thread objects, so a threading message that mentions
        // one of them means per-object tracking got cross-contaminated
        "\\((vkGetFenceStatus|vkResetFences|vkGetEventStatus)\\)|"
        "\\(vkSetEvent\\) and thread \\d+ \\(vkSetEvent\\)|\\(vkResetEvent\\) and thread \\d+ \\(vkResetEvent\\)|"
        "and another thread");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");  // Ignore any extra threading errors found beyond the first one

    RETURN_IF_SKIP(Init());

    vkt::Event shared_event(*m_device);
    vkt::Event main_event(*m_device);
    vkt::Event worker_event(*m_device);
    vkt::Fence main_fence(*m_device);
    vkt::Fence worker_fence(*m_device);

    std::atomic<bool> bailout{false};
    m_errorMonitor->SetBailout(&bailout);

    vk::SetEvent(device(), shared_event);  // warm up the tracker state from a racing thread

    std::thread thread([&]() {
        VkFence fence = worker_fence;
        for (int i = 0; i < 40000 && !bailout; i++) {
            vk::GetFenceStatus(device(), fence);
            vk::GetEventStatus(device(), worker_event);
            vk::ResetEvent(device(), shared_event);
            vk::ResetFences(device(), 1, &fence);
        }
    });
    {
        VkFence fence = main_fence;
        for (int i = 0; i < 40000 && !bailout; i++) {
            vk::GetFenceStatus(device(), fence);
            vk::GetEventStatus(device(), main_event);
            vk::SetEvent(device(), shared_event);
            vk::ResetFences(device(), 1, &fence);
        }
    }
    thread.join();

    m_errorMonitor->SetBailout(NULL);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeThreading, ReportFunctionPairImplicitCommandPool) {
    TEST_DESCRIPTION(
        "Race vkResetCommandBuffer against vkResetCommandPool. The collision is detected on the command pool that "
        "vkResetCommandBuffer uses implicitly; the message must still pair the two functions");

    m_errorMonitor->SetDesiredFailureMsgRegex(
        kErrorBit, "UNASSIGNED-Threading-MultipleThreads-Write",
        "object of type VkCommandPool is simultaneously used in current thread \\d+ \\(vkResetCommandBuffer\\) and thread \\d+ "
        "\\(vkResetCommandPool\\)|"
        "object of type VkCommandPool is simultaneously used in current thread \\d+ \\(vkResetCommandPool\\) and thread \\d+ "
        "\\(vkResetCommandBuffer\\)",
        "\\(vkResetCommandBuffer\\) and thread \\d+ \\(vkResetCommandBuffer\\)|"
        "\\(vkResetCommandPool\\) and thread \\d+ \\(vkResetCommandPool\\)|"
        "and another thread|MultipleThreads-Read");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");  // Ignore any extra threading errors found beyond the first one

    RETURN_IF_SKIP(Init());

    vkt::CommandPool pool(*m_device, m_device->graphics_queue_node_index_, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    vkt::CommandBuffer command_buffer(*m_device, pool);

    std::atomic<bool> bailout{false};
    m_errorMonitor->SetBailout(&bailout);

    vk::ResetCommandBuffer(command_buffer, 0);  // warm up the tracker state from a racing thread

    std::thread thread([&]() {
        for (int i = 0; i < 80000 && !bailout; i++) {
            vk::ResetCommandPool(device(), pool, 0);
        }
    });
    for (int i = 0; i < 80000 && !bailout; i++) {
        vk::ResetCommandBuffer(command_buffer, 0);
    }
    thread.join();

    m_errorMonitor->SetBailout(NULL);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeThreading, ReportFunctionPairStress) {
    TEST_DESCRIPTION(
        "Five threads race vkUpdateDescriptorSets on the same descriptor set; every reported pair must be "
        "(vkUpdateDescriptorSets, vkUpdateDescriptorSets)");

    m_errorMonitor->SetDesiredFailureMsgRegex(
        kErrorBit, "UNASSIGNED-Threading-MultipleThreads-Write",
        "vkUpdateDescriptorSets\\(\\): THREADING ERROR : object of type VkDescriptorSet is simultaneously used in current "
        "thread \\d+ \\(vkUpdateDescriptorSets\\) and thread \\d+ \\(vkUpdateDescriptorSets\\)",
        "and thread \\d+ \\((?!vkUpdateDescriptorSets\\))|and another thread|MultipleThreads-Read");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");  // Ignore any extra threading errors found beyond the first one

    RETURN_IF_SKIP(Init());

    OneOffDescriptorSet descriptor_set(m_device,
                                       {
                                           {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                       },
                                       0);
    vkt::Buffer buffer(*m_device, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    VkDescriptorBufferInfo buffer_info = {buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet descriptor_write = vku::InitStructHelper();
    descriptor_write.dstSet = descriptor_set.set_;
    descriptor_write.dstBinding = 0;
    descriptor_write.descriptorCount = 1;
    descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_write.pBufferInfo = &buffer_info;

    std::atomic<bool> bailout{false};
    m_errorMonitor->SetBailout(&bailout);

    vk::UpdateDescriptorSets(device(), 1, &descriptor_write, 0, NULL);  // warm up the tracker state from a racing thread

    const auto update_loop = [&]() {
        for (int i = 0; i < 40000 && !bailout; i++) {
            vk::UpdateDescriptorSets(device(), 1, &descriptor_write, 0, NULL);
        }
    };
    std::thread workers[4];
    for (auto& worker : workers) {
        worker = std::thread(update_loop);
    }
    update_loop();
    for (auto& worker : workers) {
        worker.join();
    }

    m_errorMonitor->SetBailout(NULL);
    m_errorMonitor->VerifyFound();
}

#endif  // GTEST_IS_THREADSAFE
