/*
 * Copyright (c) 2019-2023 Valve Corporation
 * Copyright (c) 2019-2023 LunarG, Inc.
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
#include "sync/sync_utils.h"
#include "state_tracker/state_tracker.h"
#include "generated/enum_flag_bits.h"

namespace sync_utils {
static constexpr uint32_t kNumPipelineStageBits = sizeof(VkPipelineStageFlags2KHR) * 8;

VkPipelineStageFlags2KHR DisabledPipelineStages(const DeviceFeatures &features) {
    VkPipelineStageFlags2KHR result = 0;
    if (!features.core.geometryShader) {
        result |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
    }
    if (!features.core.tessellationShader) {
        result |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
    }
    if (!features.conditional_rendering_features.conditionalRendering) {
        result |= VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT;
    }
    if (!features.fragment_density_map_features.fragmentDensityMap) {
        result |= VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT;
    }
    if (!features.transform_feedback_features.transformFeedback) {
        result |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    }
    if (!features.mesh_shader_features.meshShader) {
        result |= VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT;
    }
    if (!features.mesh_shader_features.taskShader) {
        result |= VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT;
    }
    if (!features.fragment_shading_rate_features.attachmentFragmentShadingRate &&
        !features.shading_rate_image_features.shadingRateImage) {
        result |= VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
    }
    // TODO: VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
    // TODO: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
    return result;
}

VkPipelineStageFlags2KHR ExpandPipelineStages(VkPipelineStageFlags2KHR stage_mask, VkQueueFlags queue_flags,
                                              const VkPipelineStageFlags2KHR disabled_feature_mask) {
    VkPipelineStageFlags2KHR expanded = stage_mask;

    if (VK_PIPELINE_STAGE_ALL_COMMANDS_BIT & stage_mask) {
        expanded &= ~VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        for (const auto &all_commands : syncAllCommandStagesByQueueFlags()) {
            if (all_commands.first & queue_flags) {
                expanded |= all_commands.second & ~disabled_feature_mask;
            }
        }
    }
    if (VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT & stage_mask) {
        expanded &= ~VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
        // Make sure we don't pull in the HOST stage from expansion, but keep it if set by the caller.
        // The syncAllCommandStagesByQueueFlags table includes HOST for all queue types since it is
        // allowed but it shouldn't be part of ALL_GRAPHICS
        expanded |=
            syncAllCommandStagesByQueueFlags().at(VK_QUEUE_GRAPHICS_BIT) & ~disabled_feature_mask & ~VK_PIPELINE_STAGE_HOST_BIT;
    }
    if (VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT_KHR & stage_mask) {
        expanded &= ~VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT_KHR;
        expanded |= VK_PIPELINE_STAGE_2_COPY_BIT_KHR | VK_PIPELINE_STAGE_2_RESOLVE_BIT_KHR | VK_PIPELINE_STAGE_2_BLIT_BIT_KHR |
                    VK_PIPELINE_STAGE_2_CLEAR_BIT_KHR;
    }
    if (VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT_KHR & stage_mask) {
        expanded &= ~VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT_KHR;
        expanded |= VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT_KHR;
    }
    if (VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT_KHR & stage_mask) {
        expanded &= ~VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT_KHR;
        expanded |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT_KHR |
                    VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT_KHR;
    }

    return expanded;
}

static const auto kShaderReadExpandBits =
    VK_ACCESS_2_UNIFORM_READ_BIT_KHR | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT_KHR | VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR;
static const auto kShaderWriteExpandBits = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;

VkAccessFlags2KHR ExpandAccessFlags(VkAccessFlags2KHR access_mask) {
    VkAccessFlags2KHR expanded = access_mask;

    if (VK_ACCESS_2_SHADER_READ_BIT_KHR & access_mask) {
        expanded = expanded & ~VK_ACCESS_2_SHADER_READ_BIT_KHR;
        expanded |= kShaderReadExpandBits;
    }

    if (VK_ACCESS_2_SHADER_WRITE_BIT_KHR & access_mask) {
        expanded = expanded & ~VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
        expanded |= kShaderWriteExpandBits;
    }

    return expanded;
}

VkAccessFlags2KHR CompatibleAccessMask(VkPipelineStageFlags2KHR stage_mask) {
    VkAccessFlags2KHR result = 0;
    stage_mask = ExpandPipelineStages(stage_mask);
    for (size_t i = 0; i < kNumPipelineStageBits; i++) {
        VkPipelineStageFlags2KHR bit = 1ULL << i;
        if (stage_mask & bit) {
            auto access_rec = syncDirectStageToAccessMask().find(bit);
            if (access_rec != syncDirectStageToAccessMask().end()) {
                result |= access_rec->second;
                continue;
            }
        }
    }

    // put the meta-access bits back on
    if (result & kShaderReadExpandBits) {
        result |= VK_ACCESS_2_SHADER_READ_BIT_KHR;
    }

    if (result & kShaderWriteExpandBits) {
        result |= VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
    }

    return result;
}

VkPipelineStageFlags2KHR RelatedPipelineStages(VkPipelineStageFlags2KHR stage_mask,
                                               const std::map<VkPipelineStageFlags2KHR, VkPipelineStageFlags2KHR> &map) {
    VkPipelineStageFlags2KHR unscanned = stage_mask;
    VkPipelineStageFlags2KHR related = 0;
    for (const auto &entry : map) {
        const auto &stage = entry.first;
        if (stage & unscanned) {
            related = related | entry.second;
            unscanned = unscanned & ~stage;
            if (!unscanned) break;
        }
    }
    return related;
}

VkPipelineStageFlags2KHR WithEarlierPipelineStages(VkPipelineStageFlags2KHR stage_mask) {
    return stage_mask | RelatedPipelineStages(stage_mask, syncLogicallyEarlierStages());
}

VkPipelineStageFlags2KHR WithLaterPipelineStages(VkPipelineStageFlags2KHR stage_mask) {
    return stage_mask | RelatedPipelineStages(stage_mask, syncLogicallyLaterStages());
}

// helper to extract the union of the stage masks in all of the barriers
ExecScopes GetGlobalStageMasks(const VkDependencyInfoKHR &dep_info) {
    ExecScopes result{};
    for (uint32_t i = 0; i < dep_info.memoryBarrierCount; i++) {
        result.src |= dep_info.pMemoryBarriers[i].srcStageMask;
        result.dst |= dep_info.pMemoryBarriers[i].dstStageMask;
    }
    for (uint32_t i = 0; i < dep_info.bufferMemoryBarrierCount; i++) {
        result.src |= dep_info.pBufferMemoryBarriers[i].srcStageMask;
        result.dst |= dep_info.pBufferMemoryBarriers[i].dstStageMask;
    }
    for (uint32_t i = 0; i < dep_info.imageMemoryBarrierCount; i++) {
        result.src |= dep_info.pImageMemoryBarriers[i].srcStageMask;
        result.dst |= dep_info.pImageMemoryBarriers[i].dstStageMask;
    }
    return result;
}

// Helpers to try to print the shortest string description of masks.
// If the bitmask doesn't use a synchronization2 specific flag, we'll
// print the old strings. There are common code paths where we need
// to print masks as strings and this makes the output less confusing
// for people not using synchronization2.
std::string StringPipelineStageFlags(VkPipelineStageFlags2KHR mask) {
    VkPipelineStageFlags sync1_mask = static_cast<VkPipelineStageFlags>(mask & AllVkPipelineStageFlagBits);
    if (sync1_mask) {
        return string_VkPipelineStageFlags(sync1_mask);
    }
    return string_VkPipelineStageFlags2(mask);
}

std::string StringAccessFlags(VkAccessFlags2KHR mask) {
    VkAccessFlags sync1_mask = static_cast<VkAccessFlags>(mask & AllVkAccessFlagBits);
    if (sync1_mask) {
        return string_VkAccessFlags(sync1_mask);
    }
    return string_VkAccessFlags2(mask);
}

ShaderStageAccesses GetShaderStageAccesses(VkShaderStageFlagBits shader_stage) {
    static const std::map<VkShaderStageFlagBits, ShaderStageAccesses> map = {
        // clang-format off
        {VK_SHADER_STAGE_VERTEX_BIT, {
            SYNC_VERTEX_SHADER_SHADER_SAMPLED_READ,
            SYNC_VERTEX_SHADER_SHADER_STORAGE_READ,
            SYNC_VERTEX_SHADER_SHADER_STORAGE_WRITE,
            SYNC_VERTEX_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, {
            SYNC_TESSELLATION_CONTROL_SHADER_SHADER_SAMPLED_READ,
            SYNC_TESSELLATION_CONTROL_SHADER_SHADER_STORAGE_READ,
            SYNC_TESSELLATION_CONTROL_SHADER_SHADER_STORAGE_WRITE,
            SYNC_TESSELLATION_CONTROL_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, {
            SYNC_TESSELLATION_EVALUATION_SHADER_SHADER_SAMPLED_READ,
            SYNC_TESSELLATION_EVALUATION_SHADER_SHADER_STORAGE_READ,
            SYNC_TESSELLATION_EVALUATION_SHADER_SHADER_STORAGE_WRITE,
            SYNC_TESSELLATION_EVALUATION_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_GEOMETRY_BIT, {
            SYNC_GEOMETRY_SHADER_SHADER_SAMPLED_READ,
            SYNC_GEOMETRY_SHADER_SHADER_STORAGE_READ,
            SYNC_GEOMETRY_SHADER_SHADER_STORAGE_WRITE,
            SYNC_GEOMETRY_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_FRAGMENT_BIT, {
            SYNC_FRAGMENT_SHADER_SHADER_SAMPLED_READ,
            SYNC_FRAGMENT_SHADER_SHADER_STORAGE_READ,
            SYNC_FRAGMENT_SHADER_SHADER_STORAGE_WRITE,
            SYNC_FRAGMENT_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_COMPUTE_BIT, {
            SYNC_COMPUTE_SHADER_SHADER_SAMPLED_READ,
            SYNC_COMPUTE_SHADER_SHADER_STORAGE_READ,
            SYNC_COMPUTE_SHADER_SHADER_STORAGE_WRITE,
            SYNC_COMPUTE_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_RAYGEN_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_ANY_HIT_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_MISS_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_INTERSECTION_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_CALLABLE_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_TASK_BIT_EXT, {
            SYNC_TASK_SHADER_EXT_SHADER_SAMPLED_READ,
            SYNC_TASK_SHADER_EXT_SHADER_STORAGE_READ,
            SYNC_TASK_SHADER_EXT_SHADER_STORAGE_WRITE,
            SYNC_TASK_SHADER_EXT_UNIFORM_READ
        }},
        {VK_SHADER_STAGE_MESH_BIT_EXT, {
            SYNC_MESH_SHADER_EXT_SHADER_SAMPLED_READ,
            SYNC_MESH_SHADER_EXT_SHADER_STORAGE_READ,
            SYNC_MESH_SHADER_EXT_SHADER_STORAGE_WRITE,
            SYNC_MESH_SHADER_EXT_UNIFORM_READ
        }},
        // clang-format on
    };
    auto it = map.find(shader_stage);
    assert(it != map.end());
    return it->second;
}

// With this flag on, use memcpy to copy fixed-size regions.
// Most compilers generate wide move instructions and treat memcpy as intrinsic (no function call).
// On x64 without any special compiler settings it's approximately 2x faster comparing to per-field
// assignment (16-byte moves vs 4/8-byte moves).
constexpr bool fast_barrier_init = true;

// Sanity check validation of C++ alignment/padding rules
constexpr size_t barrier_fields_size = 4 * sizeof(VkFlags64);  // 32 bytes
static_assert(sizeof(MemoryBarrier) == 32);
static_assert(barrier_fields_size == sizeof(MemoryBarrier));

constexpr size_t queue_family_fields_size = 2 * sizeof(uint32_t);  // 8 bytes
static_assert(sizeof(QueueFamilyBarrier) == 40);
static_assert(sizeof(MemoryBarrier) + queue_family_fields_size == sizeof(QueueFamilyBarrier));

constexpr size_t buffer_fields_size = sizeof(VkBuffer) + 2 * sizeof(VkDeviceSize);  // 24 bytes
static_assert(sizeof(BufferBarrier) == 64);
static_assert(sizeof(QueueFamilyBarrier) + buffer_fields_size == sizeof(BufferBarrier));

constexpr size_t image_fields_size = 2 * sizeof(VkImageLayout) + sizeof(VkImage) + sizeof(VkImageSubresourceRange);  // 36 bytes
static_assert(sizeof(ImageBarrier) == 80);
static_assert(sizeof(QueueFamilyBarrier) + image_fields_size + sizeof(uint32_t) /*padding*/ == sizeof(ImageBarrier));

template <typename VkBarrier>
static inline void CopyBarrier2Fields(MemoryBarrier &barrier, const VkBarrier &vk_barrier) {
    if constexpr (fast_barrier_init) {
        // two 16-byte moves on x64
        memcpy(&barrier.srcStageMask, &vk_barrier.srcStageMask, barrier_fields_size);
    } else {
        barrier.srcStageMask = vk_barrier.srcStageMask;
        barrier.srcAccessMask = vk_barrier.srcAccessMask;
        barrier.dstStageMask = vk_barrier.dstStageMask;
        barrier.dstAccessMask = vk_barrier.dstAccessMask;
    }
}

template <typename VkBarrier>
static inline void CopyBarrierFields(MemoryBarrier &barrier, const VkBarrier &vk_barrier, VkPipelineStageFlags src_stage_mask,
                                     VkPipelineStageFlags dst_stage_mask) {
    barrier.srcStageMask = src_stage_mask;
    barrier.srcAccessMask = vk_barrier.srcAccessMask;
    barrier.dstStageMask = dst_stage_mask;
    barrier.dstAccessMask = vk_barrier.dstAccessMask;
}

template <typename VkBarrier>
static inline void CopyQueueFamilyFields(QueueFamilyBarrier &barrier, const VkBarrier &vk_barrier) {
    if constexpr (fast_barrier_init) {
        // one 8-byte move on x64
        memcpy(&barrier.srcQueueFamilyIndex, &vk_barrier.srcQueueFamilyIndex, queue_family_fields_size);
    } else {
        barrier.srcQueueFamilyIndex = vk_barrier.srcQueueFamilyIndex;
        barrier.dstQueueFamilyIndex = vk_barrier.dstQueueFamilyIndex;
    }
}

template <typename VkBarrier>
static inline void CopyBufferFields(BufferBarrier &barrier, const VkBarrier &vk_barrier) {
    if constexpr (fast_barrier_init) {
        // one 16-byte move + one 8-byte move on x64
        memcpy(&barrier.buffer, &vk_barrier.buffer, buffer_fields_size);
    } else {
        barrier.buffer = vk_barrier.buffer;
        barrier.offset = vk_barrier.offset;
        barrier.size = vk_barrier.size;
    }
}

template <typename VkBarrier>
static inline void CopyImageFields(ImageBarrier &barrier, const VkBarrier &vk_barrier) {
    if constexpr (fast_barrier_init) {
        // image specific data is separated by the queue family fieds, that's why two regions
        constexpr size_t layout_fields_size = 2 * sizeof(VkImageLayout);                                 // 8 bytes
        constexpr size_t other_fields_size = image_fields_size + sizeof(uint32_t) - layout_fields_size;  // 32 bytes with padding
        memcpy(&barrier.oldLayout, &vk_barrier.oldLayout, layout_fields_size);                           // one 8-byte move
        memcpy(&barrier.image, &vk_barrier.image, other_fields_size);                                    // two 16-byte moves
    } else {
        barrier.oldLayout = vk_barrier.oldLayout;
        barrier.newLayout = vk_barrier.newLayout;
        barrier.image = vk_barrier.image;
        barrier.subresourceRange = vk_barrier.subresourceRange;
    }
}

MemoryBarrier::MemoryBarrier(const VkMemoryBarrier2 &barrier) { CopyBarrier2Fields(*this, barrier); }
MemoryBarrier::MemoryBarrier(const VkBufferMemoryBarrier2 &barrier) { CopyBarrier2Fields(*this, barrier); }
MemoryBarrier::MemoryBarrier(const VkImageMemoryBarrier2 &barrier) { CopyBarrier2Fields(*this, barrier); }

MemoryBarrier::MemoryBarrier(const VkMemoryBarrier &barrier, VkPipelineStageFlags src_stage_mask,
                             VkPipelineStageFlags dst_stage_mask) {
    CopyBarrierFields(*this, barrier, src_stage_mask, dst_stage_mask);
}
MemoryBarrier::MemoryBarrier(const VkBufferMemoryBarrier &barrier, VkPipelineStageFlags src_stage_mask,
                             VkPipelineStageFlags dst_stage_mask) {
    CopyBarrierFields(*this, barrier, src_stage_mask, dst_stage_mask);
}
MemoryBarrier::MemoryBarrier(const VkImageMemoryBarrier &barrier, VkPipelineStageFlags src_stage_mask,
                             VkPipelineStageFlags dst_stage_mask) {
    CopyBarrierFields(*this, barrier, src_stage_mask, dst_stage_mask);
}

BufferBarrier::BufferBarrier(const VkBufferMemoryBarrier2 &barrier) {
    if constexpr (fast_barrier_init) {
        // BufferBarrier repeats structure of VkBufferMemoryBarrier2 (four 16-byte moves on x64)
        memcpy(&srcStageMask, &barrier.srcStageMask, sizeof(BufferBarrier));
    } else {
        CopyBarrier2Fields(*this, barrier);
        CopyQueueFamilyFields(*this, barrier);
        CopyBufferFields(*this, barrier);
    }
}

BufferBarrier::BufferBarrier(const VkBufferMemoryBarrier &barrier, VkPipelineStageFlags src_stage_mask,
                             VkPipelineStageFlags dst_stage_mask) {
    if constexpr (fast_barrier_init) {
        CopyBarrierFields(*this, barrier, src_stage_mask, dst_stage_mask);
        // queue family fields + buffer fields (two 16-byte moves on x64)
        constexpr size_t queue_family_and_buffer_fields_size = queue_family_fields_size + buffer_fields_size;  // 32 bytes
        memcpy(&srcQueueFamilyIndex, &barrier.srcQueueFamilyIndex, queue_family_and_buffer_fields_size);
    } else {
        CopyBarrierFields(*this, barrier, src_stage_mask, dst_stage_mask);
        CopyQueueFamilyFields(*this, barrier);
        CopyBufferFields(*this, barrier);
    }
}

VulkanTypedHandle BufferBarrier::GetTypedHandle() const { return VulkanTypedHandle(buffer, kVulkanObjectTypeBuffer); }

const std::shared_ptr<const BUFFER_STATE> BufferBarrier::GetResourceState(const ValidationStateTracker &state_tracker) const {
    return state_tracker.Get<BUFFER_STATE>(buffer);
}

ImageBarrier::ImageBarrier(const VkImageMemoryBarrier2 &barrier) {
    CopyBarrier2Fields(*this, barrier);
    CopyQueueFamilyFields(*this, barrier);
    CopyImageFields(*this, barrier);
}

ImageBarrier::ImageBarrier(const VkImageMemoryBarrier &barrier, VkPipelineStageFlags src_stage_mask,
                           VkPipelineStageFlags dst_stage_mask) {
    CopyBarrierFields(*this, barrier, src_stage_mask, dst_stage_mask);
    CopyQueueFamilyFields(*this, barrier);
    CopyImageFields(*this, barrier);
}

VulkanTypedHandle ImageBarrier::GetTypedHandle() const { return VulkanTypedHandle(image, kVulkanObjectTypeImage); }

const std::shared_ptr<const IMAGE_STATE> ImageBarrier::GetResourceState(const ValidationStateTracker &state_tracker) const {
    return state_tracker.Get<IMAGE_STATE>(image);
}

}  // namespace sync_utils
