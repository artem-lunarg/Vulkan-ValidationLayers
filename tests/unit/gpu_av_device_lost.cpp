/*
 * Copyright (c) 2026 The Khronos Group Inc.
 * Copyright (c) 2026 Valve Corporation
 * Copyright (c) 2026 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// PROTOTYPE ONLY, not meant to be merged.
// The device is not lost in these tests. VK_LAYER_GPUAV_DEBUG_INJECT_DEVICE_LOST makes one GPU-AV internal command
// behave as if it returned VK_ERROR_DEVICE_LOST. The tests check when the pending GPU-AV errors are reported

#include "layer_validation_tests.h"
#include "pipeline_helper.h"
#include "descriptor_helper.h"
#include "vk_layer_config.h"

class NegativeGpuAVDeviceLost : public GpuAVBufferDeviceAddressTest {
  public:
    ~NegativeGpuAVDeviceLost() { InjectDeviceLost(""); }

    void InitDeviceLostTest();
    // Records a draw of 3 vertices. Each vertex writes 4 values through a buffer device address.
    // With out_of_bounds the first write of each vertex is before the buffer, which gives 3 errors
    void RecordDraw(vkt::CommandBuffer& cb, bool out_of_bounds);
    void InjectDeviceLost(const char* site) { SetEnvironment("VK_LAYER_GPUAV_DEBUG_INJECT_DEVICE_LOST", site); }

    const char* oob_vuid = "VUID-RuntimeSpirv-PhysicalStorageBuffer64-11819";
    std::unique_ptr<vkt::PipelineLayout> pipeline_layout;
    std::unique_ptr<CreatePipelineHelper> pipe;
    std::unique_ptr<vkt::Buffer> buffer;
};

void NegativeGpuAVDeviceLost::InitDeviceLostTest() {
    RETURN_IF_SKIP(InitGpuVUBufferDeviceAddress());
    InitRenderTarget();

    VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT, 0, 2 * sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo plci = vku::InitStructHelper();
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &push_constant_ranges;
    pipeline_layout = std::make_unique<vkt::PipelineLayout>(*m_device, plci);

    const char* shader_source = R"glsl(
            #version 450
            #extension GL_EXT_buffer_reference : enable
            layout(buffer_reference, buffer_reference_align = 16) buffer bufStruct;
            layout(push_constant) uniform ufoo {
                bufStruct data;
                int nWrites;
            } u_info;
            layout(buffer_reference, std140) buffer bufStruct {
                int a[4]; // 16 byte strides
            };
            void main() {
                for (int i=0; i < u_info.nWrites; ++i) {
                    u_info.data.a[i] = 42;
                }
            }
        )glsl";
    VkShaderObj vs(*m_device, shader_source, VK_SHADER_STAGE_VERTEX_BIT);

    pipe = std::make_unique<CreatePipelineHelper>(*this);
    pipe->shader_stages_[0] = vs.GetStageCreateInfo();
    pipe->gp_ci_.layout = *pipeline_layout;
    pipe->CreateGraphicsPipeline();

    buffer = std::make_unique<vkt::Buffer>(*m_device, 64, 0, vkt::device_address);
}

void NegativeGpuAVDeviceLost::RecordDraw(vkt::CommandBuffer& cb, bool out_of_bounds) {
    cb.Begin();
    cb.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, *pipe);
    const VkDeviceAddress address = buffer->Address();
    VkDeviceAddress push_constants[2] = {out_of_bounds ? address - 16 : address, 4};
    vk::CmdPushConstants(cb, *pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_constants), push_constants);
    vk::CmdDraw(cb, 3, 1, 0, 0);
    cb.EndRenderPass();
    cb.End();
}

TEST_F(NegativeGpuAVDeviceLost, InternalPreSubmit) {
    TEST_DESCRIPTION("Loss detected by a GPU-AV pre submit reports the errors of an earlier submission");
    RETURN_IF_SKIP(InitDeviceLostTest());

    vkt::CommandBuffer cb2(*m_device, m_command_pool);
    RecordDraw(m_command_buffer, true);
    RecordDraw(cb2, false);

    // No wait, so this submission stays pending in the layer and its errors are not read
    m_default_queue->Submit(m_command_buffer);

    InjectDeviceLost("presubmit");
    m_errorMonitor->SetDesiredError(oob_vuid, 3);
    m_default_queue->Submit(cb2);
    // Nothing asked the queue thread to retire the first submission, so the errors were reported by this vkQueueSubmit
    m_errorMonitor->VerifyFound();

    InjectDeviceLost("");
    m_default_queue->Wait();
}

TEST_F(NegativeGpuAVDeviceLost, InternalPostSubmit) {
    TEST_DESCRIPTION("Loss detected by a GPU-AV post submit reports the errors of the submission that was just made");
    RETURN_IF_SKIP(InitDeviceLostTest());

    RecordDraw(m_command_buffer, true);

    InjectDeviceLost("barrier");
    m_errorMonitor->SetDesiredError(oob_vuid, 3);
    m_default_queue->Submit(m_command_buffer);
    m_errorMonitor->VerifyFound();

    InjectDeviceLost("");
    m_default_queue->Wait();
}

TEST_F(NegativeGpuAVDeviceLost, CompletedNonFinalBatch) {
    TEST_DESCRIPTION("A batch that completed but waits for the last batch of its submit to be read is reported");
    SetTargetApiVersion(VK_API_VERSION_1_2);
    AddRequiredFeature(vkt::Feature::timelineSemaphore);
    RETURN_IF_SKIP(InitDeviceLostTest());

    vkt::CommandBuffer cb2(*m_device, m_command_pool);
    vkt::CommandBuffer cb3(*m_device, m_command_pool);
    RecordDraw(m_command_buffer, true);
    RecordDraw(cb2, false);
    RecordDraw(cb3, false);

    vkt::Semaphore semaphore(*m_device, VK_SEMAPHORE_TYPE_TIMELINE);
    const uint64_t signal_value = 1;
    VkTimelineSemaphoreSubmitInfo timeline_info = vku::InitStructHelper();
    timeline_info.signalSemaphoreValueCount = 1;
    timeline_info.pSignalSemaphoreValues = &signal_value;

    VkSubmitInfo submits[2];
    submits[0] = vku::InitStructHelper(&timeline_info);
    submits[0].commandBufferCount = 1;
    submits[0].pCommandBuffers = &m_command_buffer.handle();
    submits[0].signalSemaphoreCount = 1;
    submits[0].pSignalSemaphores = &semaphore.handle();
    submits[1] = vku::InitStructHelper();
    submits[1].commandBufferCount = 1;
    submits[1].pCommandBuffers = &cb2.handle();
    vk::QueueSubmit(*m_default_queue, 2, submits, VK_NULL_HANDLE);

    // Retires the first batch only. GPU-AV keeps it until the second batch is retired
    semaphore.Wait(signal_value, kWaitTimeout);

    InjectDeviceLost("presubmit");
    m_errorMonitor->SetDesiredError(oob_vuid, 3);
    m_default_queue->Submit(cb3);
    m_errorMonitor->VerifyFound();

    InjectDeviceLost("");
    m_default_queue->Wait();
}

TEST_F(NegativeGpuAVDeviceLost, QueueThread) {
    TEST_DESCRIPTION("Loss detected by the queue thread still reports the errors of the submission it retires");
    RETURN_IF_SKIP(InitDeviceLostTest());

    RecordDraw(m_command_buffer, true);
    m_default_queue->Submit(m_command_buffer);

    InjectDeviceLost("retire");
    m_errorMonitor->SetDesiredError(oob_vuid, 3);
    m_default_queue->Wait();
    m_errorMonitor->VerifyFound();
    InjectDeviceLost("");
}

TEST_F(NegativeGpuAVDeviceLost, QueueThreadThenSubmit) {
    TEST_DESCRIPTION("After the queue thread detected the loss, the next submit reports the errors of the other submissions");
    RETURN_IF_SKIP(InitDeviceLostTest());

    vkt::CommandBuffer cb2(*m_device, m_command_pool);
    vkt::CommandBuffer cb3(*m_device, m_command_pool);
    RecordDraw(cb2, false);
    RecordDraw(m_command_buffer, true);
    // Submitted after the loss, when GPU-AV does not prepare submissions anymore.
    // The device is not lost for real, so this command buffer must not run instrumented shaders
    cb3.Begin();
    cb3.End();

    vkt::Fence fence(*m_device);
    m_default_queue->Submit(cb2, fence);
    m_default_queue->Submit(m_command_buffer);

    // The queue thread retires the first submission and detects the loss. The second submission is not retired
    InjectDeviceLost("retire");
    fence.Wait(kWaitTimeout);
    InjectDeviceLost("");

    // The second submission completes. The layer retires it without GPU-AV, so its errors are still not read
    m_default_queue->Wait();

    m_errorMonitor->SetDesiredError(oob_vuid, 3);
    m_default_queue->Submit(cb3);
    m_errorMonitor->VerifyFound();

    m_default_queue->Wait();
}

TEST_F(NegativeGpuAVDeviceLost, OtherErrorFamily) {
    TEST_DESCRIPTION("Errors that are not decoded after a loss are reported as one warning for their command");
    RETURN_IF_SKIP(InitDeviceLostTest());

    // Reads past the 4 bytes bound to the uniform buffer
    const char* fs_source = R"glsl(
        #version 450
        layout(location=0) out vec4 x;
        layout(set=0, binding=0) uniform readonly foo { int x; int y; } bar;
        void main() {
           x = vec4(bar.x, bar.y, 0, 1);
        }
    )glsl";
    VkShaderObj vs(*m_device, kVertexDrawPassthroughGlsl, VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderObj fs(*m_device, fs_source, VK_SHADER_STAGE_FRAGMENT_BIT);

    OneOffDescriptorSet descriptor_set(m_device, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr}});
    const vkt::PipelineLayout uniform_pipeline_layout(*m_device, {&descriptor_set.layout_});
    vkt::Buffer uniform_buffer(*m_device, 4, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    descriptor_set.WriteDescriptorBufferInfo(0, uniform_buffer, 0, 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    descriptor_set.UpdateDescriptorSets();

    CreatePipelineHelper uniform_pipe(*this);
    uniform_pipe.shader_stages_ = {vs.GetStageCreateInfo(), fs.GetStageCreateInfo()};
    uniform_pipe.gp_ci_.layout = uniform_pipeline_layout;
    uniform_pipe.CreateGraphicsPipeline();

    m_command_buffer.Begin();
    m_command_buffer.BeginRenderPass(m_renderPassBeginInfo);
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, uniform_pipe);
    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, uniform_pipeline_layout, 0, 1,
                              &descriptor_set.set_, 0, nullptr);
    vk::CmdDraw(m_command_buffer, 3, 1, 0, 0);
    m_command_buffer.EndRenderPass();
    m_command_buffer.End();

    InjectDeviceLost("barrier");
    // One warning for the loss, one for the draw whose errors are not decoded
    m_errorMonitor->SetDesiredWarning("WARNING-GPU-Assisted-Validation", 2);
    m_default_queue->Submit(m_command_buffer);
    m_errorMonitor->VerifyFound();

    InjectDeviceLost("");
    m_default_queue->Wait();
}
