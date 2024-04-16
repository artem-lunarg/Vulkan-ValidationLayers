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

struct PositiveSyncValTimelineSemaphore : public VkSyncValTest {};

TEST_F(PositiveSyncValTimelineSemaphore, WaitInitialValue) {
    TEST_DESCRIPTION("Waiting for the initial value should result in no wait and be successful");
    RETURN_IF_SKIP(InitTimelineSemaphore());
    vkt::Semaphore semaphore(*m_device, VK_SEMAPHORE_TYPE_TIMELINE, 1);
    m_default_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::wait, semaphore, 1);
    m_default_queue->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, SignalThenWait) {
    TEST_DESCRIPTION("Signal then wait for signaled value");
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
    m_default_queue->Submit2WithTimelineSemaphore(command_buffer2, vkt::wait, semaphore, 1);
    m_default_queue->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, SignalThenWaitTwoQueues) {
    TEST_DESCRIPTION("Signal then wait for signaled value");
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
    m_second_queue->Submit2WithTimelineSemaphore(m_second_command_buffer, vkt::wait, semaphore, 1);
    m_device->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, SignalThenWaitSmallerValue) {
    TEST_DESCRIPTION("Signal a value then wait for a smaller value");
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
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 2);
    m_default_queue->Submit2WithTimelineSemaphore(command_buffer2, vkt::wait, semaphore, 1);
    m_default_queue->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, SignalThenWaitSmallerValueTwoQueues) {
    TEST_DESCRIPTION("Signal a value then wait for a smaller value");
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
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 2);
    m_second_queue->Submit2WithTimelineSemaphore(m_second_command_buffer, vkt::wait, semaphore, 1);
    m_device->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, SignalThenWaitNonDefaultStage) {
    TEST_DESCRIPTION("Signal and wait restrict synchronization scopes");
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
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 3, VK_PIPELINE_STAGE_2_COPY_BIT);
    m_default_queue->Submit2WithTimelineSemaphore(command_buffer2, vkt::wait, semaphore, 1, VK_PIPELINE_STAGE_2_COPY_BIT);
    m_default_queue->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, SignalThenWaitNonDefaultStageTwoQueues) {
    TEST_DESCRIPTION("Signal and wait restrict synchronization scopes");
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
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 3, VK_PIPELINE_STAGE_2_COPY_BIT);
    m_second_queue->Submit2WithTimelineSemaphore(m_second_command_buffer, vkt::wait, semaphore, 1, VK_PIPELINE_STAGE_2_COPY_BIT);
    m_device->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, WaitBeforeSignal) {
    TEST_DESCRIPTION("Wait on the first queue then signal from a different queue");
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
    m_second_queue->Submit2WithTimelineSemaphore(m_second_command_buffer, vkt::signal, semaphore, 1);
    m_device->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, WaitSmallerValueBeforeSignal) {
    TEST_DESCRIPTION("Wait for a value on the first queue then signal a larger value from a different queue");
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
    m_second_queue->Submit2WithTimelineSemaphore(m_second_command_buffer, vkt::signal, semaphore, 2);
    m_device->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, WaitBeforeSignalNonDefaultStage) {
    TEST_DESCRIPTION("Signal and wait restrict synchronization scopes");
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
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::wait, semaphore, 1, VK_PIPELINE_STAGE_2_COPY_BIT);
    m_second_queue->Submit2WithTimelineSemaphore(m_second_command_buffer, vkt::signal, semaphore, 3, VK_PIPELINE_STAGE_2_COPY_BIT);
    m_device->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, WaitLatestSignal) {
    TEST_DESCRIPTION("");
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
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 1, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
    m_default_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::signal, semaphore, 2);  // includes all stages
    m_default_queue->Submit2WithTimelineSemaphore(command_buffer2, vkt::wait, semaphore, 2);
    m_device->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, WaitLatestSignalTwoQueues) {
    TEST_DESCRIPTION("");
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
    m_default_queue->Submit2WithTimelineSemaphore(m_command_buffer, vkt::signal, semaphore, 1, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
    m_default_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::signal, semaphore, 2);  // includes all stages
    m_second_queue->Submit2WithTimelineSemaphore(m_second_command_buffer, vkt::wait, semaphore, 2);
    m_device->Wait();
}

TEST_F(PositiveSyncValTimelineSemaphore, ResolveWait) {
    TEST_DESCRIPTION("Two queues resolve wait-before-signal for one another");
    RETURN_IF_SKIP(InitTimelineSemaphore());

    if (!m_second_queue) {
        GTEST_SKIP() << "Two queues are needed";
    }

    vkt::Semaphore first_queue_wait(*m_device, VK_SEMAPHORE_TYPE_TIMELINE);
    vkt::Semaphore second_queue_wait(*m_device, VK_SEMAPHORE_TYPE_TIMELINE);

    m_second_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::wait, second_queue_wait, 1);
    m_second_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::signal, first_queue_wait, 1);
    m_default_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::signal, second_queue_wait, 1);
    m_default_queue->Submit2WithTimelineSemaphore(vkt::no_cmd, vkt::wait, first_queue_wait, 1);
    m_device->Wait();
}
