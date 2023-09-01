/*
 * Copyright (c) 2019-2021, 2023 Valve Corporation
 * Copyright (c) 2019-2021, 2023 LunarG, Inc.
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

#pragma once
#include "generated/sync_validation_types.h"
#include "generated/vk_object_types.h"
#include <vulkan/vulkan.h>
#include <string>

// Remove Windows trojan macro that prevents usage of this name in any scope of the program.
// For example, nested namespace type sync_utils::MemoryBarrier won't compile because of this.
#if defined(VK_USE_PLATFORM_WIN32_KHR) && defined(MemoryBarrier)
#undef MemoryBarrier
#endif

struct DeviceFeatures;
class ValidationStateTracker;
class BUFFER_STATE;
class IMAGE_STATE;

namespace sync_utils {

static constexpr VkQueueFlags kAllQueueTypes = (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);

VkPipelineStageFlags2KHR DisabledPipelineStages(const DeviceFeatures& features);

// Expand all pipeline stage bits. If queue_flags and disabled_feature_mask is provided, the expansion of ALL_COMMANDS_BIT
// and ALL_GRAPHICS_BIT will be limited to what is supported.
VkPipelineStageFlags2KHR ExpandPipelineStages(VkPipelineStageFlags2KHR stage_mask, VkQueueFlags queue_flags = kAllQueueTypes,
                                              const VkPipelineStageFlags2KHR disabled_feature_mask = 0);

VkAccessFlags2KHR ExpandAccessFlags(VkAccessFlags2KHR access_mask);

VkAccessFlags2KHR CompatibleAccessMask(VkPipelineStageFlags2KHR stage_mask);

VkPipelineStageFlags2KHR WithEarlierPipelineStages(VkPipelineStageFlags2KHR stage_mask);

VkPipelineStageFlags2KHR WithLaterPipelineStages(VkPipelineStageFlags2KHR stage_mask);

std::string StringPipelineStageFlags(VkPipelineStageFlags2KHR mask);

std::string StringAccessFlags(VkAccessFlags2KHR mask);

struct ExecScopes {
    VkPipelineStageFlags2KHR src;
    VkPipelineStageFlags2KHR dst;
};
ExecScopes GetGlobalStageMasks(const VkDependencyInfoKHR& dep_info);

struct ShaderStageAccesses {
    SyncStageAccessIndex sampled_read;
    SyncStageAccessIndex storage_read;
    SyncStageAccessIndex storage_write;
    SyncStageAccessIndex uniform_read;
};
ShaderStageAccesses GetShaderStageAccesses(VkShaderStageFlagBits shader_stage);

struct MemoryBarrier {
    MemoryBarrier() = default;
    explicit MemoryBarrier(const VkMemoryBarrier2& barrier);
    explicit MemoryBarrier(const VkBufferMemoryBarrier2& barrier);
    explicit MemoryBarrier(const VkImageMemoryBarrier2& barrier);
    MemoryBarrier(const VkMemoryBarrier& barrier, VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask);
    MemoryBarrier(const VkBufferMemoryBarrier& barrier, VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask);
    MemoryBarrier(const VkImageMemoryBarrier& barrier, VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask);

    VkPipelineStageFlags2 srcStageMask;
    VkAccessFlags2 srcAccessMask;
    VkPipelineStageFlags2 dstStageMask;
    VkAccessFlags2 dstAccessMask;
};

// Objects of this type are not created directly (there are no queue family barriers).
// If queue family information is needed, but not the buffer/image part of the barrier
// structure, then resouce barriers can be downcasted to this type. This allows to
// declare queue family related api that works both with buffer and image barriers.
struct QueueFamilyBarrier : MemoryBarrier {
    uint32_t srcQueueFamilyIndex;
    uint32_t dstQueueFamilyIndex;
};

struct BufferBarrier : QueueFamilyBarrier {
    explicit BufferBarrier(const VkBufferMemoryBarrier2& barrier);
    BufferBarrier(const VkBufferMemoryBarrier& barrier, VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask);

    VulkanTypedHandle GetTypedHandle() const;
    const std::shared_ptr<const BUFFER_STATE> GetResourceState(const ValidationStateTracker& state_tracker) const;

    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
};

struct ImageBarrier : QueueFamilyBarrier {
    explicit ImageBarrier(const VkImageMemoryBarrier2& barrier);
    ImageBarrier(const VkImageMemoryBarrier& barrier, VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask);

    VulkanTypedHandle GetTypedHandle() const;
    const std::shared_ptr<const IMAGE_STATE> GetResourceState(const ValidationStateTracker& state_tracker) const;

    VkImageLayout oldLayout;
    VkImageLayout newLayout;
    VkImage image;
    VkImageSubresourceRange subresourceRange;
    // show 4-byte padding (compilers have to add this because of 8-byte image field)
    uint32_t padding;
};

}  // namespace sync_utils
