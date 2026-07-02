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

// The collision tests below hammer an external-synchronization violation from two threads and rely on
// their Start/Finish windows overlapping, so they are statistical: expected to report on average, a
// single run may miss. The ThreadTimeoutHelper watchdog additionally turns any reintroduced deadlock
// into a failure instead of a hang. Matching positive (must-not-report) tests live in
// threading_positive.cpp.

// Two threads reset the same fence: both take StartWriteObject(fence) -> write-vs-write collision.
TEST_F(NegativeThreading, ResetFenceCollision) {
    TEST_DESCRIPTION("Two threads resetting the same fence should raise a threading error");
    m_errorMonitor->SetDesiredError("THREADING ERROR");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");  // tolerate extra threading errors after the first

    RETURN_IF_SKIP(Init());

    vkt::Fence fence(*m_device);
    const VkFence fence_handle = fence;
    const VkDevice dev = device();

    std::atomic<bool> bailout{false};
    m_errorMonitor->SetBailout(&bailout);

    ThreadTimeoutHelper timeout(2);
    auto worker = [&]() {
        auto guard = timeout.ThreadGuard();
        for (int i = 0; i < 100000 && !bailout.load(); i++) {
            vk::ResetFences(dev, 1, &fence_handle);
        }
    };
    std::thread t0(worker);
    std::thread t1(worker);
    if (!timeout.WaitForThreads(60)) ADD_FAILURE() << "Worker threads did not finish in time";
    t0.join();
    t1.join();

    m_errorMonitor->SetBailout(nullptr);
    m_errorMonitor->VerifyFound();
}

// One thread queries the fence status (StartReadObject), another resets it (StartWriteObject):
// read-vs-write collision, exercising HandleErrorOnRead in addition to HandleErrorOnWrite.
TEST_F(NegativeThreading, FenceStatusVsResetCollision) {
    TEST_DESCRIPTION("Querying a fence status while another thread resets the same fence should raise a threading error");
    m_errorMonitor->SetDesiredError("THREADING ERROR");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");

    RETURN_IF_SKIP(Init());

    vkt::Fence fence(*m_device);
    const VkFence fence_handle = fence;
    const VkDevice dev = device();

    std::atomic<bool> bailout{false};
    m_errorMonitor->SetBailout(&bailout);

    ThreadTimeoutHelper timeout(2);
    auto reader = [&]() {
        auto guard = timeout.ThreadGuard();
        for (int i = 0; i < 100000 && !bailout.load(); i++) {
            vk::GetFenceStatus(dev, fence_handle);
        }
    };
    auto writer = [&]() {
        auto guard = timeout.ThreadGuard();
        for (int i = 0; i < 100000 && !bailout.load(); i++) {
            vk::ResetFences(dev, 1, &fence_handle);
        }
    };
    std::thread t0(reader);
    std::thread t1(writer);
    if (!timeout.WaitForThreads(60)) ADD_FAILURE() << "Worker threads did not finish in time";
    t0.join();
    t1.join();

    m_errorMonitor->SetBailout(nullptr);
    m_errorMonitor->VerifyFound();
}

// Two threads wait for the same queue to idle: both take StartWriteObject(queue).
TEST_F(NegativeThreading, QueueWaitIdleCollision) {
    TEST_DESCRIPTION("Two threads calling vkQueueWaitIdle on the same queue should raise a threading error");
    m_errorMonitor->SetDesiredError("THREADING ERROR");
    m_errorMonitor->SetAllowedFailureMsg("THREADING ERROR");

    RETURN_IF_SKIP(Init());

    const VkQueue queue = m_default_queue->handle();

    std::atomic<bool> bailout{false};
    m_errorMonitor->SetBailout(&bailout);

    ThreadTimeoutHelper timeout(2);
    auto worker = [&]() {
        auto guard = timeout.ThreadGuard();
        for (int i = 0; i < 40000 && !bailout.load(); i++) {
            vk::QueueWaitIdle(queue);
        }
    };
    std::thread t0(worker);
    std::thread t1(worker);
    if (!timeout.WaitForThreads(60)) ADD_FAILURE() << "Worker threads did not finish in time";
    t0.join();
    t1.join();

    m_errorMonitor->SetBailout(nullptr);
    m_errorMonitor->VerifyFound();
}

#endif  // GTEST_IS_THREADSAFE
