/* Copyright (c) 2024 The Khronos Group Inc.
 * Copyright (c) 2024 Valve Corporation
 * Copyright (c) 2024 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../framework/layer_validation_tests.h"

struct NegativeSyncValTimelineSemaphore : public VkSyncValTest {};

TEST_F(NegativeSyncValTimelineSemaphore, WaitInitialValue) {
    TEST_DESCRIPTION("Wait on the initial value finishes before signal. This results in WAW hazard");
    RETURN_IF_SKIP(InitTimelineSemaphore());

    vkt::Buffer buffer_a(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    vkt::Buffer buffer_b(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_command_buffer.begin();
    m_command_buffer.Copy(buffer_a, buffer_b);
    m_command_buffer.end();
    vkt::CommandBuffer command_buffer2(*m_device, m_command_pool);
    command_buffer2.begin();
    command_buffer2.Copy(buffer_a, buffer_b);
    command_buffer2.end();

    vkt::Semaphore semaphore(*m_device, VK_SEMAPHORE_TYPE_TIMELINE);
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 1);

    // Wait for the initial value results in no wait
    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    m_default_queue->Submit2WithTimelineSemaphore(command_buffer2, vkt::wait, semaphore, 0);
    m_errorMonitor->VerifyFound();
    m_default_queue->Wait();
}

TEST_F(NegativeSyncValTimelineSemaphore, WaitInitialValueTwoQueues) {
    TEST_DESCRIPTION("Wait on the initial value finishes before signal. This results in WAW hazard");
    RETURN_IF_SKIP(InitTimelineSemaphore());

    if (!m_second_queue) {
        GTEST_SKIP() << "Two queues are needed";
    }
    vkt::Buffer buffer_a(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    vkt::Buffer buffer_b(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_command_buffer.begin();
    m_command_buffer.Copy(buffer_a, buffer_b);
    m_command_buffer.end();
    m_second_command_buffer.begin();
    m_second_command_buffer.Copy(buffer_a, buffer_b);
    m_second_command_buffer.end();

    vkt::Semaphore semaphore(*m_device, VK_SEMAPHORE_TYPE_TIMELINE);
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 1);

    // Wait for the initial value results in no wait
    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-RACING-WRITE");
    m_second_queue->Submit2WithTimelineSemaphore(m_second_command_buffer, vkt::wait, semaphore, 0);
    m_errorMonitor->VerifyFound();
    m_device->Wait();
}

TEST_F(NegativeSyncValTimelineSemaphore, SignalThenWaitStageMismatch) {
    TEST_DESCRIPTION("Hazard due to semaphore stage mask mismatch in signal then wait scenario");
    RETURN_IF_SKIP(InitTimelineSemaphore());

    vkt::Buffer buffer_a(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    vkt::Buffer buffer_b(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_command_buffer.begin();
    m_command_buffer.Copy(buffer_a, buffer_b);
    m_command_buffer.end();
    vkt::CommandBuffer command_buffer2(*m_device, m_command_pool);
    command_buffer2.begin();
    command_buffer2.Copy(buffer_a, buffer_b);
    command_buffer2.end();

    vkt::Semaphore semaphore(*m_device, VK_SEMAPHORE_TYPE_TIMELINE);
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 2, VK_PIPELINE_STAGE_2_CLEAR_BIT);
    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    m_default_queue->Submit2WithTimelineSemaphore(command_buffer2, vkt::wait, semaphore, 1, VK_PIPELINE_STAGE_2_COPY_BIT);
    m_errorMonitor->VerifyFound();
    m_default_queue->Wait();
}

TEST_F(NegativeSyncValTimelineSemaphore, WaitBeforeSignal) {
    TEST_DESCRIPTION("Signal from the second queue resolves wait on the main queue");
    RETURN_IF_SKIP(InitTimelineSemaphore());

    if (!m_second_queue) {
        GTEST_SKIP() << "Two queues are needed";
    }
    vkt::Buffer buffer_a(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    vkt::Buffer buffer_b(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_command_buffer.begin();
    m_command_buffer.Copy(buffer_a, buffer_b);
    m_command_buffer.end();
    m_second_command_buffer.begin();
    m_second_command_buffer.Copy(buffer_a, buffer_b);
    m_second_command_buffer.end();

    vkt::Semaphore semaphore(*m_device, VK_SEMAPHORE_TYPE_TIMELINE);

    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::wait, semaphore, 1);
    m_second_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::signal, semaphore, 1);

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-RACING-WRITE");
    m_second_queue->Submit2(m_second_command_buffer);
    m_errorMonitor->VerifyFound();
    m_device->Wait();
}

TEST_F(NegativeSyncValTimelineSemaphore, WaitBeforeSignalEmptyWaitScope) {
    TEST_DESCRIPTION("Hazard due to semaphore stage mask mismatch in wait before signal scenario");
    RETURN_IF_SKIP(InitTimelineSemaphore());

    if (!m_second_queue) {
        GTEST_SKIP() << "Two queues are needed";
    }
    vkt::Buffer buffer_a(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    vkt::Buffer buffer_b(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_command_buffer.begin();
    m_command_buffer.Copy(buffer_a, buffer_b);
    m_command_buffer.end();
    m_second_command_buffer.begin();
    m_second_command_buffer.Copy(buffer_a, buffer_b);
    m_second_command_buffer.end();

    vkt::Semaphore semaphore(*m_device, VK_SEMAPHORE_TYPE_TIMELINE);

    // Buffer copy starts immediately because wait stage mask does not cover copy operation
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::wait, semaphore, 1, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    m_second_queue->Submit2WithTimelineSemaphore(m_second_command_buffer, vkt::signal, semaphore, 1);
    m_errorMonitor->VerifyFound();

    // Unblock the first queue (the previous call due to error did not reach Record phase)
    m_second_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::signal, semaphore, 1);
    m_device->Wait();
}

TEST_F(NegativeSyncValTimelineSemaphore, WaitNotLatestSignal) {
    TEST_DESCRIPTION("Wait on the signal before the last one");
    RETURN_IF_SKIP(InitTimelineSemaphore());

    vkt::Buffer buffer_a(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    vkt::Buffer buffer_b(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_command_buffer.begin();
    m_command_buffer.Copy(buffer_a, buffer_b);
    m_command_buffer.end();
    vkt::CommandBuffer command_buffer2(*m_device, m_command_pool);
    command_buffer2.begin();
    command_buffer2.Copy(buffer_a, buffer_b);
    command_buffer2.end();

    vkt::Semaphore semaphore(*m_device, VK_SEMAPHORE_TYPE_TIMELINE);

    // Signal stage mask does not protect buffer write access
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 1, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

    // Submit one more signal on the same queue to guard all stages.
    m_default_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::signal, semaphore, 2);

    // Check that waiting on the older signal (value=1) does not accidentally result in waiting
    // on the latest signal (value=2). If, due to bug, the implementation waits on the latest
    // signal then no hazard happens, because signal 2 synchronizes all stages.
    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    m_default_queue->Submit2WithTimelineSemaphore(command_buffer2, vkt::wait, semaphore, 1);
    m_errorMonitor->VerifyFound();

    m_device->Wait();
}

TEST_F(NegativeSyncValTimelineSemaphore, WaitNotLatestSignalTwoQueues) {
    TEST_DESCRIPTION("Wait on the signal before the last one from a different queue");
    RETURN_IF_SKIP(InitTimelineSemaphore());

    if (!m_second_queue) {
        GTEST_SKIP() << "Two queues are needed";
    }
    vkt::Buffer buffer_a(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    vkt::Buffer buffer_b(*m_device, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_command_buffer.begin();
    m_command_buffer.Copy(buffer_a, buffer_b);
    m_command_buffer.end();
    m_second_command_buffer.begin();
    m_second_command_buffer.Copy(buffer_a, buffer_b);
    m_second_command_buffer.end();

    vkt::Semaphore semaphore(*m_device, VK_SEMAPHORE_TYPE_TIMELINE);

    // Signal stage mask does not protect buffer write access
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 1, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

    // Submit one more signal on the same queue to guard all stages.
    m_default_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::signal, semaphore, 2);

    // Check that waiting on the older signal (value=1) does not accidentally result in waiting
    // on the latest signal (value=2). If, due to bug, the implementation waits on the latest
    // signal then no hazard happens, because signal 2 synchronizes all stages.
    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    m_second_queue->Submit2WithTimelineSemaphore(m_second_command_buffer, vkt::wait, semaphore, 1);
    m_errorMonitor->VerifyFound();

    m_device->Wait();
}
