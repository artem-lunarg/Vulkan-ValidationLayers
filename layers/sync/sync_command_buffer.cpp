/*
 * Copyright (c) 2019-2026 Valve Corporation
 * Copyright (c) 2019-2026 LunarG, Inc.
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

#include <vulkan/utility/vk_format_utils.h>
#include "sync/sync_command_buffer.h"
#include "error_message/error_location.h"
#include "sync/sync_image.h"
#include "sync/sync_replay.h"
#include "sync/sync_reporting.h"
#include "sync/sync_validation.h"
#include "state_tracker/descriptor_sets.h"
#include "state_tracker/image_state.h"
#include "state_tracker/buffer_state.h"
#include "state_tracker/event_state.h"
#include "state_tracker/ray_tracing_state.h"
#include "state_tracker/render_pass_state.h"
#include "state_tracker/shader_module.h"
#include "state_tracker/pipeline_state.h"
#include "utils/image_utils.h"
#include "utils/math_utils.h"
#include "utils/text_utils.h"

namespace syncval {

constexpr VkImageAspectFlags kColorAspects =
    VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT | VK_IMAGE_ASPECT_PLANE_2_BIT;

constexpr SyncAccessIndex kResolveRead = SYNC_COLOR_ATTACHMENT_OUTPUT_COLOR_ATTACHMENT_READ;
constexpr SyncAccessIndex kResolveWrite = SYNC_COLOR_ATTACHMENT_OUTPUT_COLOR_ATTACHMENT_WRITE;
constexpr SyncOrdering kColorResolveOrder = SyncOrdering::kColorAttachment;

// Although depth resolve runs on the color attachment output stage and uses color accesses, depth accesses
// still participate in the ordering. That's why using raster and not only color attachment ordering
constexpr SyncOrdering kDepthStencilResolveOrder = SyncOrdering::kRaster;

constexpr SyncOrdering kStoreOrder = SyncOrdering::kRaster;

struct ShaderStageAccesses {
    SyncAccessIndex sampled_read;
    SyncAccessIndex storage_read;
    SyncAccessIndex storage_write;
    SyncAccessIndex uniform_read;
    SyncAccessIndex acceleration_structure_read;
};

// TODO: generate me
static ShaderStageAccesses GetShaderStageAccesses(VkShaderStageFlagBits shader_stage) {
    static const vvl::unordered_map<VkShaderStageFlagBits, ShaderStageAccesses> map = {
        // clang-format off
        {VK_SHADER_STAGE_VERTEX_BIT, {
            SYNC_VERTEX_SHADER_SHADER_SAMPLED_READ,
            SYNC_VERTEX_SHADER_SHADER_STORAGE_READ,
            SYNC_VERTEX_SHADER_SHADER_STORAGE_WRITE,
            SYNC_VERTEX_SHADER_UNIFORM_READ,
            SYNC_VERTEX_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, {
            SYNC_TESSELLATION_CONTROL_SHADER_SHADER_SAMPLED_READ,
            SYNC_TESSELLATION_CONTROL_SHADER_SHADER_STORAGE_READ,
            SYNC_TESSELLATION_CONTROL_SHADER_SHADER_STORAGE_WRITE,
            SYNC_TESSELLATION_CONTROL_SHADER_UNIFORM_READ,
            SYNC_TESSELLATION_CONTROL_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, {
            SYNC_TESSELLATION_EVALUATION_SHADER_SHADER_SAMPLED_READ,
            SYNC_TESSELLATION_EVALUATION_SHADER_SHADER_STORAGE_READ,
            SYNC_TESSELLATION_EVALUATION_SHADER_SHADER_STORAGE_WRITE,
            SYNC_TESSELLATION_EVALUATION_SHADER_UNIFORM_READ,
            SYNC_TESSELLATION_EVALUATION_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_GEOMETRY_BIT, {
            SYNC_GEOMETRY_SHADER_SHADER_SAMPLED_READ,
            SYNC_GEOMETRY_SHADER_SHADER_STORAGE_READ,
            SYNC_GEOMETRY_SHADER_SHADER_STORAGE_WRITE,
            SYNC_GEOMETRY_SHADER_UNIFORM_READ,
            SYNC_GEOMETRY_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_FRAGMENT_BIT, {
            SYNC_FRAGMENT_SHADER_SHADER_SAMPLED_READ,
            SYNC_FRAGMENT_SHADER_SHADER_STORAGE_READ,
            SYNC_FRAGMENT_SHADER_SHADER_STORAGE_WRITE,
            SYNC_FRAGMENT_SHADER_UNIFORM_READ,
            SYNC_FRAGMENT_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_COMPUTE_BIT, {
            SYNC_COMPUTE_SHADER_SHADER_SAMPLED_READ,
            SYNC_COMPUTE_SHADER_SHADER_STORAGE_READ,
            SYNC_COMPUTE_SHADER_SHADER_STORAGE_WRITE,
            SYNC_COMPUTE_SHADER_UNIFORM_READ,
            SYNC_COMPUTE_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_RAYGEN_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ,
            SYNC_RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_ANY_HIT_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ,
            SYNC_RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ,
            SYNC_RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_MISS_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ,
            SYNC_RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_INTERSECTION_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ,
            SYNC_RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_CALLABLE_BIT_KHR, {
            SYNC_RAY_TRACING_SHADER_SHADER_SAMPLED_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_READ,
            SYNC_RAY_TRACING_SHADER_SHADER_STORAGE_WRITE,
            SYNC_RAY_TRACING_SHADER_UNIFORM_READ,
            SYNC_RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_TASK_BIT_EXT, {
            SYNC_TASK_SHADER_EXT_SHADER_SAMPLED_READ,
            SYNC_TASK_SHADER_EXT_SHADER_STORAGE_READ,
            SYNC_TASK_SHADER_EXT_SHADER_STORAGE_WRITE,
            SYNC_TASK_SHADER_EXT_UNIFORM_READ,
            SYNC_TASK_SHADER_EXT_ACCELERATION_STRUCTURE_READ,
        }},
        {VK_SHADER_STAGE_MESH_BIT_EXT, {
            SYNC_MESH_SHADER_EXT_SHADER_SAMPLED_READ,
            SYNC_MESH_SHADER_EXT_SHADER_STORAGE_READ,
            SYNC_MESH_SHADER_EXT_SHADER_STORAGE_WRITE,
            SYNC_MESH_SHADER_EXT_UNIFORM_READ,
            SYNC_MESH_SHADER_EXT_ACCELERATION_STRUCTURE_READ,
        }},
        // clang-format on
    };
    auto it = map.find(shader_stage);
    assert(it != map.end());
    return it->second;
}

static AccessRange MakeRangeForVertexData(VkDeviceSize offset, uint32_t first_vertex, uint32_t vertex_count,
                                          const VertexBindingState& vertex_binding) {
    uint32_t element_size = 0;
    for (const auto& [_, vertex_attrib] : vertex_binding.locations) {
        element_size = std::max(element_size, vertex_attrib.desc.offset + GetVertexInputFormatSize(vertex_attrib.desc.format));
    }
    const VkDeviceSize range_start = offset + (first_vertex * vertex_binding.desc.stride);
    VkDeviceSize range_size = 0;
    if (vertex_count > 0) {
        // Take into account stride between elements but not after the last element.
        range_size = (vertex_count - 1) * vertex_binding.desc.stride + element_size;
    }
    return MakeRange(range_start, range_size);
}

static AccessRange MakeRangeForIndexData(VkDeviceSize offset, uint32_t first_index, uint32_t index_count, uint32_t index_size) {
    const VkDeviceSize range_start = offset + (first_index * index_size);
    const VkDeviceSize range_size = index_count * index_size;
    return MakeRange(range_start, range_size);
}

static AccessRange MakeRange(const vvl::BufferView& buf_view_state) {
    return MakeRange(*buf_view_state.buffer_state.get(), buf_view_state.create_info.offset, buf_view_state.create_info.range);
}

static SyncAccessIndex GetSyncStageAccessIndexsByDescriptorSet(VkDescriptorType descriptor_type,
                                                               const spirv::ResourceInterfaceVariable& variable,
                                                               VkShaderStageFlagBits stage_flag) {
    if (!variable.IsAccessed()) {
        return SYNC_ACCESS_INDEX_NONE;
    }
    if (descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) {
        assert(stage_flag == VK_SHADER_STAGE_FRAGMENT_BIT);
        return SYNC_FRAGMENT_SHADER_INPUT_ATTACHMENT_READ;
    }
    const auto stage_accesses = GetShaderStageAccesses(stage_flag);

    if (descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
        return stage_accesses.uniform_read;
    }
    if (descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR ||
        descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV ||
        descriptor_type == VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV) {
        return stage_accesses.acceleration_structure_read;
    }

    // If the desriptorSet is writable, we don't need to care SHADER_READ. SHADER_WRITE is enough.
    // Because if write hazard happens, read hazard might or might not happen.
    // But if write hazard doesn't happen, read hazard is impossible to happen.
    if (variable.IsWrittenTo()) {
        return stage_accesses.storage_write;
    } else if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
               descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
               descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) {
        return stage_accesses.sampled_read;
    } else {
        if (variable.IsImage() && !variable.IsImageReadFrom()) {
            // only image descriptor was accessed, not the image data
            return SYNC_ACCESS_INDEX_NONE;
        }
        return stage_accesses.storage_read;
    }
}

SyncEnvironment::SyncEnvironment(const SyncValidator& validator, VkQueueFlags queue_flags, QueueId queue_id,
                                 VulkanTypedHandle handle, SyncEventsContext& events_context,
                                 const ResourceUsageInfoProvider& usage_info_provider)
    : validator(validator),
      queue_flags(queue_flags),
      queue_id(queue_id),
      handle(handle),
      events_context(events_context),
      usage_info_provider(usage_info_provider) {}

CommandBufferContext::CommandBufferContext(const SyncValidator& sync_validator, VkQueueFlags queue_flags, VulkanTypedHandle handle)
    : sync_state_(sync_validator),
      error_messages_(sync_validator.error_messages_),
      access_log_(std::make_shared<AccessLog>()),
      cbs_referenced_(std::make_shared<CommandBufferSet>()),
      command_number_(0),
      reset_count_(0),
      cb_access_context_(sync_validator),
      current_context_(&cb_access_context_),
      events_context_(),
      environment_(sync_validator, queue_flags, kQueueIdInvalid, handle, events_context_, *this),
      render_pass_contexts_(),
      current_renderpass_context_() {}

CommandBufferContext::CommandBufferContext(SyncValidator& sync_validator, vvl::CommandBuffer* cb_state)
    : CommandBufferContext(sync_validator, cb_state->GetQueueFlags(), cb_state->Handle()) {
    cb_state_ = cb_state;
    sync_state_.stats.AddCommandBufferContext();
}

// NOTE: Make sure the proxy doesn't outlive from, as the proxy is pointing directly to access contexts owned by from.
CommandBufferContext::CommandBufferContext(const CommandBufferContext& from, AsProxyContext dummy)
    : CommandBufferContext(from.sync_state_, from.cb_state_->GetQueueFlags(), from.cb_state_->Handle()) {
    // Copy only the needed fields out of from for a temporary, proxy command buffer context
    cb_state_ = from.cb_state_;
    access_log_ = std::make_shared<AccessLog>(*from.access_log_);  // potentially large, but no choice given tagging lookup.
    command_number_ = from.command_number_;
    reset_count_ = from.reset_count_;

    handles_ = from.handles_;
    sync_state_.stats.AddHandleRecord((uint32_t)from.handles_.size());

    const AccessContext& from_context = from.GetCurrentAccessContext();

    // Construct a fully resolved single access context out of from
    cb_access_context_.ResolveFromContextRecursePrev(from_context);
    // The proxy has flatten the current render pass context (if any), but the async contexts are needed for hazard detection
    cb_access_context_.ImportAsyncContexts(from_context);

    events_context_ = from.events_context_;

    // We don't want to copy the full render_pass_context_ history just for the proxy.
    sync_state_.stats.AddCommandBufferContext();
}

CommandBufferContext::~CommandBufferContext() {
    sync_state_.stats.RemoveCommandBufferContext();
    sync_state_.stats.RemoveHandleRecord((uint32_t)handles_.size());
}

void CommandBufferContext::Reset() {
    access_log_ = std::make_shared<AccessLog>();
    cbs_referenced_ = std::make_shared<CommandBufferSet>();
    if (cb_state_) {
        cbs_referenced_->push_back(cb_state_->shared_from_this());
    }
    replay_entries_.clear();
    recorded_commands_.clear();
    command_data_ = {};
    command_number_ = 0;
    reset_count_++;

    sync_state_.stats.RemoveHandleRecord((uint32_t)handles_.size());
    handles_.clear();

    current_command_tag_ = vvl::kNoIndex32;
    cb_access_context_.Reset();
    render_pass_contexts_.clear();
    current_context_ = &cb_access_context_;
    current_renderpass_context_ = nullptr;
    events_context_.Clear();
    dynamic_rendering_info_.reset();
}

bool CommandBufferContext::ValidateBeginRendering(const ErrorObject& error_obj, BeginRenderingCmdState& cmd_state) const {
    bool skip = false;
    const DynamicRenderingInfo& info = cmd_state.GetRenderingInfo();

    // Load operations do not happen when resuming
    if (info.info.flags & VK_RENDERING_RESUMING_BIT) {
        return skip;
    }

    // Need to hazard detect load operations vs. the attachment views
    for (size_t i = 0; i < info.attachments.size(); i++) {
        const auto& attachment = info.attachments[i];
        const SyncAccessIndex load_index = attachment.GetLoadUsage();
        if (load_index == SYNC_ACCESS_INDEX_NONE) {
            continue;
        }

        const AttachmentAccess attachment_access = GetAttachmentAccess(attachment.GetOrdering(), AttachmentAccessType::LoadOp);
        ImageRangeGen range_gen = attachment.GetRangeGen(info.info.viewMask);
        const HazardResult hazard = GetCbAccessContext().DetectAttachmentHazard(range_gen, load_index, attachment_access);
        if (hazard.IsHazard()) {
            LogObjectList objlist(cb_state_->Handle(), attachment.view->Handle());

            std::ostringstream ss;
            ss << vvl::String(vvl::Field::pRenderingInfo) << ".";
            ss << attachment.GetLocation(error_obj.location, uint32_t(i)).Fields();
            ss << " (" << sync_state_.FormatHandle(attachment.view->Handle());
            ss << ", loadOp " << string_VkAttachmentLoadOp(attachment.info.loadOp) << ")";
            std::string resource_description = ss.str();

            const std::string error = sync_state_.error_messages_.BeginRenderingError(hazard, *this, error_obj.location.function,
                                                                                      resource_description, attachment.info.loadOp);
            skip |= sync_state_.SyncError(hazard.Hazard(), objlist, error_obj.location.function, error);
            if (skip) {
                break;
            }
        }
    }
    return skip;
}

ResourceAccessCommand CommandBufferContext::MakeBeginRenderingAccessCommand(const DynamicRenderingInfo& rendering_info) const {
    ResourceAccessCommand command;
    if ((rendering_info.info.flags & VK_RENDERING_RESUMING_BIT) != 0) {
        return command;
    }
    for (const auto& attachment : rendering_info.attachments) {
        const SyncAccessIndex load_index = attachment.GetLoadUsage();
        if (!attachment.view || load_index == SYNC_ACCESS_INDEX_NONE) {
            continue;
        }
        ResourceAccessCommand::ImageViewAccess access;
        access.image_view = attachment.view;
        access.access_index = load_index;
        access.use_render_area = rendering_info.info.viewMask == 0;
        access.offset = CastTo3D(rendering_info.info.renderArea.offset);
        access.extent = CastTo3D(rendering_info.info.renderArea.extent);
        access.view_mask = rendering_info.info.viewMask;
        access.aspect_mask = attachment.type == AttachmentType::kDepth     ? VK_IMAGE_ASPECT_DEPTH_BIT
                             : attachment.type == AttachmentType::kStencil ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                                           : 0;
        access.attachment_access = GetAttachmentAccess(attachment.GetOrdering(), AttachmentAccessType::LoadOp);
        access.message_type = "BeginRenderingError";
        command.accesses.emplace_back(std::move(access));
    }
    return command;
}

void CommandBufferContext::RecordBeginRendering(BeginRenderingCmdState& cmd_state, const Location& loc) {
    const auto tag = NextCommandTag(loc.function);
    const DynamicRenderingInfo& info = cmd_state.GetRenderingInfo();
    auto command = MakeBeginRenderingAccessCommand(info);
    command.Record(*this, tag, sync_state_.syncval_settings.IsRecordTimeValidationEnabled());
    AddRecordedCommand(tag, std::move(command));
    dynamic_rendering_info_ = std::move(cmd_state.info);
}

bool CommandBufferContext::ValidateEndRendering(const ErrorObject& error_obj) const {
    bool skip = false;

    // Only validate resolve and store if not suspending (as specified by BeginRendering)
    if (!dynamic_rendering_info_ || (dynamic_rendering_info_->info.flags & VK_RENDERING_SUSPENDING_BIT) != 0) {
        return skip;
    }

    for (uint32_t i = 0; i < (uint32_t)dynamic_rendering_info_->attachments.size(); i++) {
        const auto& attachment = dynamic_rendering_info_->attachments[i];

        auto attachment_description = [this, &error_obj, &attachment, i](const auto& view, std::ostringstream& ss) {
            ss << vvl::String(vvl::Field::pRenderingInfo) << ".";
            ss << attachment.GetLocation(error_obj.location, uint32_t(i)).Fields();
            ss << " (" << sync_state_.FormatHandle(view->Handle());
        };

        // The logic about whether to resolve is embedded in the Attachment constructor
        if (attachment.resolve_gen) {
            const bool is_color = attachment.type == AttachmentType::kColor;
            const SyncOrdering kResolveOrder = is_color ? kColorResolveOrder : kDepthStencilResolveOrder;

            const AttachmentAccess resolve_read_access = GetAttachmentAccess(kResolveOrder, AttachmentAccessType::ResolveRead);
            ImageRangeGen view_gen = attachment.GetRangeGen(dynamic_rendering_info_->info.viewMask);
            HazardResult hazard = current_context_->DetectAttachmentHazard(view_gen, kResolveRead, resolve_read_access);
            if (hazard.IsHazard()) {
                LogObjectList objlist(cb_state_->Handle(), attachment.view->Handle());

                std::ostringstream ss;
                attachment_description(attachment.view, ss);
                ss << ", resolveMode " << string_VkResolveModeFlagBits(attachment.info.resolveMode) << ")";
                const std::string resource_description = ss.str();

                const std::string error = sync_state_.error_messages_.EndRenderingResolveError(
                    hazard, *this, error_obj.location.function, resource_description, attachment.info.resolveMode, false);
                skip |= sync_state_.SyncError(hazard.Hazard(), objlist, error_obj.location.function, error);
                if (skip) {
                    break;
                }
            }

            const AttachmentAccess resolve_write_access = GetAttachmentAccess(kResolveOrder, AttachmentAccessType::ResolveWrite);
            ImageRangeGen resolve_gen = *attachment.resolve_gen;
            hazard = current_context_->DetectAttachmentHazard(resolve_gen, kResolveWrite, resolve_write_access);
            if (hazard.IsHazard()) {
                LogObjectList objlist(cb_state_->Handle(), attachment.resolve_view->Handle());

                std::ostringstream ss;
                attachment_description(attachment.resolve_view, ss);
                ss << ", resolveMode " << string_VkResolveModeFlagBits(attachment.info.resolveMode) << ")";
                const std::string resource_description = ss.str();

                const std::string error = sync_state_.error_messages_.EndRenderingResolveError(
                    hazard, *this, error_obj.location.function, resource_description, attachment.info.resolveMode, true);
                skip |= sync_state_.SyncError(hazard.Hazard(), objlist, error_obj.location.function, error);
                if (skip) {
                    break;
                }
            }
        }

        const SyncAccessIndex store_access = attachment.GetStoreUsage();
        if (store_access != SYNC_ACCESS_INDEX_NONE) {
            const AttachmentAccess attachment_access = GetAttachmentAccess(kStoreOrder, AttachmentAccessType::StoreOp);
            ImageRangeGen view_gen = attachment.GetRangeGen(dynamic_rendering_info_->info.viewMask);

            HazardResult hazard = current_context_->DetectAttachmentHazard(view_gen, store_access, attachment_access);
            if (hazard.IsHazard()) {
                LogObjectList objlist(cb_state_->Handle(), attachment.view->Handle());

                std::ostringstream ss;
                attachment_description(attachment.view, ss);
                ss << ", storeOp " << string_VkAttachmentStoreOp(attachment.info.storeOp) << ")";
                const std::string resource_description = ss.str();

                const std::string error = sync_state_.error_messages_.EndRenderingStoreError(
                    hazard, *this, error_obj.location.function, resource_description, attachment.info.storeOp);
                skip |= sync_state_.SyncError(hazard.Hazard(), objlist, error_obj.location.function, error);
                if (skip) {
                    break;
                }
            }
        }
    }
    return skip;
}

ResourceAccessCommand CommandBufferContext::MakeEndRenderingAccessCommand() const {
    ResourceAccessCommand command;
    if (!dynamic_rendering_info_ || (dynamic_rendering_info_->info.flags & VK_RENDERING_SUSPENDING_BIT) != 0) {
        return command;
    }

    auto add_access = [&](const std::shared_ptr<const vvl::ImageView>& view, SyncAccessIndex access_index,
                          const AttachmentAccess& attachment_access, AttachmentType type, uint32_t view_mask,
                          const char* message_type) {
        if (!view || access_index == SYNC_ACCESS_INDEX_NONE) {
            return;
        }
        ResourceAccessCommand::ImageViewAccess access;
        access.image_view = view;
        access.access_index = access_index;
        access.use_render_area = view_mask == 0;
        access.offset = CastTo3D(cb_state_->render_area.offset);
        access.extent = CastTo3D(cb_state_->render_area.extent);
        access.view_mask = view_mask;
        access.aspect_mask = type == AttachmentType::kDepth     ? VK_IMAGE_ASPECT_DEPTH_BIT
                             : type == AttachmentType::kStencil ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                                : 0;
        access.attachment_access = attachment_access;
        access.message_type = message_type;
        command.accesses.emplace_back(std::move(access));
    };

    for (const auto& attachment : dynamic_rendering_info_->attachments) {
        if (attachment.resolve_gen) {
            const SyncOrdering resolve_order =
                attachment.type == AttachmentType::kColor ? kColorResolveOrder : kDepthStencilResolveOrder;
            add_access(attachment.view, kResolveRead, GetAttachmentAccess(resolve_order, AttachmentAccessType::ResolveRead),
                       attachment.type, dynamic_rendering_info_->info.viewMask, "EndRenderingResolveError");
            add_access(attachment.resolve_view, kResolveWrite,
                       GetAttachmentAccess(resolve_order, AttachmentAccessType::ResolveWrite), attachment.type, 0,
                       "EndRenderingResolveError");
        }
        add_access(attachment.view, attachment.GetStoreUsage(), GetAttachmentAccess(kStoreOrder, AttachmentAccessType::StoreOp),
                   attachment.type, dynamic_rendering_info_->info.viewMask, "EndRenderingStoreError");
    }
    return command;
}

void CommandBufferContext::RecordEndRendering(const RecordObject& record_obj) {
    if (!dynamic_rendering_info_) {
        return;
    }
    if ((dynamic_rendering_info_->info.flags & VK_RENDERING_SUSPENDING_BIT) != 0) {
        dynamic_rendering_info_.reset();
        return;
    }
    const auto store_tag = NextCommandTag(record_obj.location.function, SubCommandType::kStoreOp);
    auto command = MakeEndRenderingAccessCommand();
    command.Record(*this, store_tag, sync_state_.syncval_settings.IsRecordTimeValidationEnabled());
    AddRecordedCommand(store_tag, std::move(command));
    current_render_pass_instance_id_++;
    dynamic_rendering_info_.reset();
}

ResourceAccessCommand CommandBufferContext::MakeDispatchDrawDescriptorAccessCommand(VkPipelineBindPoint pipelineBindPoint) const {
    ResourceAccessCommand command;
    if (!sync_state_.syncval_settings.shader_accesses_heuristic) {
        return command;
    }

    const auto& last_bound_state = cb_state_->lastBound[ConvertToVvlBindPoint(pipelineBindPoint)];
    const vvl::Pipeline* pipe = last_bound_state.pipeline_state;
    const std::vector<LastBound::DescriptorSetSlot>& ds_slots = last_bound_state.ds_slots;
    if (!pipe) {
        return command;
    }
    const auto pipeline = sync_state_.Get<vvl::Pipeline>(pipe->VkHandle());

    using DescriptorClass = vvl::DescriptorClass;
    using BufferDescriptor = vvl::BufferDescriptor;
    using ImageDescriptor = vvl::ImageDescriptor;
    using TexelDescriptor = vvl::TexelDescriptor;

    for (const auto& stage_state : pipe->stage_states) {
        if (stage_state.GetStage() == VK_SHADER_STAGE_FRAGMENT_BIT && pipe->RasterizationDisabled()) {
            continue;
        } else if (!stage_state.HasSpirv()) {
            continue;
        }
        for (const auto& variable : stage_state.entrypoint->resource_interface_variables) {
            if (variable.decorations.set >= ds_slots.size()) {
                continue;
            }
            const auto& ds_slot = ds_slots[variable.decorations.set];
            const auto* descriptor_set = ds_slot.ds_state.get();
            if (!descriptor_set) {
                continue;
            }
            const auto binding = descriptor_set->GetBinding(variable.decorations.binding);
            if (!binding || binding->count > 1) {
                continue;
            }
            const auto descriptor_set_state = sync_state_.Get<vvl::DescriptorSet>(descriptor_set->VkHandle());
            const VkDescriptorType descriptor_type = binding->type;
            const SyncAccessIndex sync_index =
                GetSyncStageAccessIndexsByDescriptorSet(descriptor_type, variable, stage_state.GetStage());

            auto make_descriptor_info = [&](ResourceAccessCommand::DescriptorResourceType resource_type,
                                            const VulkanTypedHandle& resource_handle,
                                            VkImageLayout image_layout = VK_IMAGE_LAYOUT_UNDEFINED) {
                ResourceAccessCommand::DescriptorInfo info;
                info.pipeline = pipeline;
                info.descriptor_set = descriptor_set_state;
                info.resource_handle = resource_handle;
                info.resource_type = resource_type;
                info.set = variable.decorations.set;
                info.descriptor_type = descriptor_type;
                info.binding = variable.decorations.binding;
                info.array_element = 0;
                info.stage = stage_state.GetStage();
                info.image_layout = image_layout;
                return info;
            };

            const auto* descriptor = binding->GetDescriptor(0);
            switch (descriptor->GetClass()) {
                case DescriptorClass::ImageSampler:
                case DescriptorClass::Image: {
                    const auto* image_descriptor = static_cast<const ImageDescriptor*>(descriptor);
                    if (image_descriptor->Invalid()) {
                        continue;
                    }
                    const auto* image_view = image_descriptor->GetImageViewState();
                    if (!image_view || image_view->is_depth_sliced) {
                        continue;
                    }
                    ResourceAccessCommand::ImageViewAccess access;
                    access.image_view = sync_state_.Get<vvl::ImageView>(image_view->VkHandle());
                    access.access_index = sync_index;
                    access.tag_handle = image_view->image_state->Handle();
                    access.descriptor_info = make_descriptor_info(ResourceAccessCommand::DescriptorResourceType::kImage,
                                                                  image_view->Handle(), image_descriptor->GetImageLayout());
                    if (sync_index == SYNC_FRAGMENT_SHADER_INPUT_ATTACHMENT_READ) {
                        access.use_render_area = true;
                        access.offset = CastTo3D(cb_state_->render_area.offset);
                        access.extent = CastTo3D(cb_state_->render_area.extent);
                        access.attachment_access = GetAttachmentAccess(SyncOrdering::kRaster);
                    }
                    command.accesses.emplace_back(std::move(access));
                    break;
                }
                case DescriptorClass::TexelBuffer: {
                    const auto* texel_descriptor = static_cast<const TexelDescriptor*>(descriptor);
                    if (texel_descriptor->Invalid()) {
                        continue;
                    }
                    const auto* buffer_view = texel_descriptor->GetBufferViewState();
                    const auto* buffer = buffer_view->buffer_state.get();
                    ResourceAccessCommand::BufferAccess access;
                    access.buffer = sync_state_.Get<vvl::Buffer>(buffer->VkHandle());
                    access.access_index = sync_index;
                    access.range = MakeRange(*buffer_view);
                    access.tag_handle = buffer_view->Handle();
                    access.descriptor_info =
                        make_descriptor_info(ResourceAccessCommand::DescriptorResourceType::kBuffer, buffer_view->Handle());
                    command.accesses.emplace_back(std::move(access));
                    break;
                }
                case DescriptorClass::GeneralBuffer: {
                    const auto* buffer_descriptor = static_cast<const BufferDescriptor*>(descriptor);
                    if (buffer_descriptor->Invalid()) {
                        continue;
                    }
                    VkDeviceSize offset = buffer_descriptor->GetOffset();
                    if (vvl::IsDynamicDescriptor(descriptor_type)) {
                        const uint32_t dynamic_offset_index = descriptor_set->GetDynamicOffsetIndexFromBinding(binding->binding);
                        if (dynamic_offset_index >= ds_slot.dynamic_offsets.size()) {
                            continue;
                        }
                        offset += ds_slot.dynamic_offsets[dynamic_offset_index];
                    }
                    const auto* buffer = buffer_descriptor->GetBufferState();
                    ResourceAccessCommand::BufferAccess access;
                    access.buffer = sync_state_.Get<vvl::Buffer>(buffer->VkHandle());
                    access.access_index = sync_index;
                    access.range = MakeRange(*buffer, offset, buffer_descriptor->GetRange());
                    access.tag_handle = buffer->Handle();
                    access.descriptor_info =
                        make_descriptor_info(ResourceAccessCommand::DescriptorResourceType::kBuffer, buffer->Handle());
                    command.accesses.emplace_back(std::move(access));
                    break;
                }
                case DescriptorClass::AccelerationStructure: {
                    const auto* accel_descriptor = static_cast<const vvl::AccelerationStructureDescriptor*>(descriptor);
                    if (accel_descriptor->Invalid()) {
                        continue;
                    }
                    const vvl::AccelerationStructureKHR* accel = accel_descriptor->GetAccelerationStructureStateKHR();
                    if (!accel) {
                        continue;
                    }
                    const vvl::BufferAndOffset as_buffer = accel->GetFirstValidBuffer(cb_state_->dev_data);
                    if (!as_buffer) {
                        continue;
                    }
                    ResourceAccessCommand::BufferAccess access;
                    access.buffer = sync_state_.Get<vvl::Buffer>(as_buffer.state->VkHandle());
                    access.access_index = sync_index;
                    access.range = MakeRange(*as_buffer.state, as_buffer.offset, accel->GetSize());
                    access.tag_handle = accel->Handle();
                    access.descriptor_info = make_descriptor_info(
                        ResourceAccessCommand::DescriptorResourceType::kAccelerationStructure, accel->Handle());
                    command.accesses.emplace_back(std::move(access));
                    break;
                }
                default:
                    break;
            }
        }
    }
    return command;
}

ResourceAccessCommand CommandBufferContext::MakeDrawVertexAccessCommand(uint32_t vertex_count, uint32_t first_vertex) const {
    ResourceAccessCommand command;
    const auto* pipe = cb_state_->GetLastBoundGraphics().pipeline_state;
    if (!pipe) {
        return command;
    }
    const auto pipeline = sync_state_.Get<vvl::Pipeline>(pipe->VkHandle());

    const auto& binding_buffers = cb_state_->current_vertex_buffer_binding_info;
    const auto& vertex_bindings = pipe->IsDynamic(CB_DYNAMIC_STATE_VERTEX_INPUT_EXT)
                                      ? cb_state_->dynamic_state_value.vertex_bindings
                                      : pipe->vertex_input_state->bindings;
    for (const auto& [_, binding_state] : vertex_bindings) {
        if (binding_state.desc.inputRate != VK_VERTEX_INPUT_RATE_VERTEX) {
            continue;
        }
        const vvl::VertexBufferBinding* vertex_buffer = vvl::Find(binding_buffers, binding_state.desc.binding);
        if (!vertex_buffer) {
            continue;
        }
        const auto buffer = sync_state_.Get<vvl::Buffer>(vertex_buffer->Buffer());
        if (!buffer) {
            continue;
        }
        ResourceAccessCommand::BufferAccess access;
        access.buffer = buffer;
        access.pipeline = pipeline;
        access.access_index = SYNC_VERTEX_ATTRIBUTE_INPUT_VERTEX_ATTRIBUTE_READ;
        access.range = MakeRangeForVertexData(vertex_buffer->BufferOffset(), first_vertex, vertex_count, binding_state);
        access.tag_handle = buffer->Handle();
        access.resource_name = "vertex ";
        command.accesses.emplace_back(std::move(access));
    }
    return command;
}

ResourceAccessCommand CommandBufferContext::MakeDrawVertexIndexAccessCommand(uint32_t index_count, uint32_t first_index) const {
    ResourceAccessCommand command;
    const auto& index_binding = cb_state_->index_buffer_binding;
    const auto buffer = sync_state_.Get<vvl::Buffer>(index_binding.Buffer());
    if (!buffer) {
        return command;
    }
    ResourceAccessCommand::BufferAccess access;
    access.buffer = buffer;
    if (const auto* pipe = cb_state_->GetLastBoundGraphics().pipeline_state) {
        access.pipeline = sync_state_.Get<vvl::Pipeline>(pipe->VkHandle());
    }
    access.access_index = SYNC_INDEX_INPUT_INDEX_READ;
    access.range =
        MakeRangeForIndexData(index_binding.BufferOffset(), first_index, index_count, IndexTypeByteSize(index_binding.index_type));
    access.tag_handle = buffer->Handle();
    access.resource_name = "index ";
    command.accesses.emplace_back(std::move(access));
    return command;
}

ResourceAccessCommand CommandBufferContext::MakeDrawAttachmentAccessCommand() const {
    ResourceAccessCommand command;
    if (current_renderpass_context_) {
        return current_renderpass_context_->MakeDrawSubpassAttachmentAccessCommand(*cb_state_);
    }
    if (!dynamic_rendering_info_) {
        return command;
    }
    const auto& last_bound_state = cb_state_->GetLastBoundGraphics();
    const auto* pipe = last_bound_state.pipeline_state;
    if (!pipe || pipe->RasterizationDisabled()) {
        return command;
    }

    const DynamicRenderingInfo& info = *dynamic_rendering_info_;
    auto add_attachment = [&](const DynamicRenderingInfo::Attachment& attachment, uint32_t attachment_index,
                              SyncAccessIndex access_index, SyncOrdering ordering) {
        if (!attachment.view) {
            return;
        }
        ResourceAccessCommand::ImageViewAccess access;
        access.image_view = attachment.view;
        access.access_index = access_index;
        access.use_render_area = info.info.viewMask == 0;
        access.offset = CastTo3D(cb_state_->render_area.offset);
        access.extent = CastTo3D(cb_state_->render_area.extent);
        access.view_mask = info.info.viewMask;
        access.aspect_mask = attachment.type == AttachmentType::kDepth     ? VK_IMAGE_ASPECT_DEPTH_BIT
                             : attachment.type == AttachmentType::kStencil ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                                           : 0;
        access.attachment_access = GetAttachmentAccess(ordering);
        access.tag_handle = attachment.view->Handle();
        access.message_type = "DynamicRenderingAttachmentError";
        access.attachment_index = attachment_index;
        access.error_location =
            attachment.type == AttachmentType::kColor   ? ResourceAccessCommand::ImageViewAccess::ErrorLocation::kColorAttachment
            : attachment.type == AttachmentType::kDepth ? ResourceAccessCommand::ImageViewAccess::ErrorLocation::kDepthAttachment
                                                        : ResourceAccessCommand::ImageViewAccess::ErrorLocation::kStencilAttachment;
        command.accesses.emplace_back(std::move(access));
    };

    for (const uint32_t output_location : pipe->fs_writable_output_location_list) {
        if (output_location >= info.info.colorAttachmentCount) {
            continue;
        }
        const auto& attachment = info.attachments[output_location];
        if (attachment.IsWriteable(last_bound_state)) {
            add_attachment(attachment, output_location, SYNC_COLOR_ATTACHMENT_OUTPUT_COLOR_ATTACHMENT_WRITE,
                           SyncOrdering::kColorAttachment);
        }
    }
    for (size_t i = info.info.colorAttachmentCount; i < info.attachments.size(); ++i) {
        const auto& attachment = info.attachments[i];
        if (attachment.IsWriteable(last_bound_state)) {
            add_attachment(attachment, static_cast<uint32_t>(i), SYNC_LATE_FRAGMENT_TESTS_DEPTH_STENCIL_ATTACHMENT_WRITE,
                           SyncOrdering::kDepthStencilAttachment);
        }
    }
    return command;
}

VkImageAspectFlags CommandBufferContext::GetAttachmentAspectsToClear(VkImageAspectFlags clear_aspect_mask,
                                                                     const vvl::ImageView& attachment_view) const {
    // Check if clear request is valid.
    const bool clear_color = (clear_aspect_mask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
    const bool clear_depth = (clear_aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
    const bool clear_stencil = (clear_aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    if (!clear_color && !clear_depth && !clear_stencil) {
        return 0;  // nothing to clear
    }
    if (clear_color && (clear_depth || clear_stencil)) {
        return 0;  // according to spec it's not allowed
    }

    // Color aspects to clear
    if (clear_color) {
        // The image view aspect mask is used only for color attachment.
        // For depth/stencil attachment, it is ignored according to the spec.
        const VkImageAspectFlags view_aspect_mask = attachment_view.normalized_subresource_range.aspectMask;
        return view_aspect_mask & kColorAspects;
    }

    // Depth-stencil aspects to clear
    bool has_depth_attachment = false;
    bool has_stencil_attachment = false;
    if (dynamic_rendering_info_) {
        has_depth_attachment = dynamic_rendering_info_->info.pDepthAttachment != nullptr;
        has_stencil_attachment = dynamic_rendering_info_->info.pStencilAttachment != nullptr;
    } else if (current_renderpass_context_) {
        const auto& rp_create_info = current_renderpass_context_->GetRenderPassState()->create_info;
        const auto& subpass = rp_create_info.pSubpasses[current_renderpass_context_->GetCurrentSubpass()];
        if (subpass.pDepthStencilAttachment) {
            const uint32_t attachment = subpass.pDepthStencilAttachment->attachment;
            if (attachment < rp_create_info.attachmentCount) {
                const VkFormat ds_format = rp_create_info.pAttachments[attachment].format;
                has_depth_attachment = vkuFormatHasDepth(ds_format);
                has_stencil_attachment = vkuFormatHasStencil(ds_format);
            }
        }
    }
    VkImageAspectFlags ds_aspects_to_clear = VK_IMAGE_ASPECT_NONE;
    if (clear_depth && has_depth_attachment) {
        ds_aspects_to_clear |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (clear_stencil && has_stencil_attachment) {
        ds_aspects_to_clear |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return ds_aspects_to_clear;
}

static std::optional<VkImageSubresourceRange> RestrictSubresourceRangeToClearLayers(
    const VkImageSubresourceRange& normalized_subresource_range, uint32_t clear_first_layer, uint32_t clear_layer_count) {
    // Contract of this function
    assert(normalized_subresource_range.layerCount != VK_REMAINING_ARRAY_LAYERS);
    // According to spec
    assert(clear_layer_count != VK_REMAINING_ARRAY_LAYERS);

    const uint32_t first = std::max(normalized_subresource_range.baseArrayLayer, clear_first_layer);
    const uint32_t last_range = normalized_subresource_range.baseArrayLayer + normalized_subresource_range.layerCount;
    const uint32_t last_clear = clear_first_layer + clear_layer_count;
    const uint32_t last = std::min(last_range, last_clear);

    if (first >= last) {
        return {};
    }

    std::optional<VkImageSubresourceRange> result = normalized_subresource_range;
    result->baseArrayLayer = first;
    result->layerCount = last - first;
    return result;
}

std::optional<CommandBufferContext::ClearAttachmentInfo> CommandBufferContext::GetClearAttachmentInfo(
    const VkClearAttachment& clear_attachment, uint32_t clear_first_layer, uint32_t clear_layer_count) const {
    const vvl::ImageView* attachment_view = nullptr;
    if (current_renderpass_context_) {
        attachment_view = current_renderpass_context_->GetClearAttachmentView(clear_attachment);
    } else if (dynamic_rendering_info_) {
        attachment_view = dynamic_rendering_info_->GetClearAttachmentView(clear_attachment);
    }
    if (!attachment_view) {
        return {};
    }

    const VkImageAspectFlags aspects_to_clear = GetAttachmentAspectsToClear(clear_attachment.aspectMask, *attachment_view);
    if (!aspects_to_clear) {
        return {};
    }

    std::optional<VkImageSubresourceRange> subresource_range =
        RestrictSubresourceRangeToClearLayers(attachment_view->normalized_subresource_range, clear_first_layer, clear_layer_count);
    if (!subresource_range.has_value()) {
        return {};
    }
    subresource_range->aspectMask = aspects_to_clear;
    return ClearAttachmentInfo{*attachment_view, *subresource_range};
}

bool CommandBufferContext::ValidateClearAttachment(const Location& loc, const VkClearAttachment& clear_attachment,
                                                   uint32_t clear_rect_index, const VkClearRect& clear_rect) const {
    bool skip = false;

    const auto optional_info = GetClearAttachmentInfo(clear_attachment, clear_rect.baseArrayLayer, clear_rect.layerCount);
    if (!optional_info) {
        return skip;
    }
    const ClearAttachmentInfo& info = *optional_info;
    const VkImageSubresourceRange subresource_range = info.subresource_range;
    const VkImageAspectFlags aspects_to_clear = subresource_range.aspectMask;
    const uint32_t view_mask = GetViewMask();
    const ImageSubState& sub_state = SubState(*info.attachment_view.image_state);

    // NOTE: when we teach ImageRangeGen to work with view masks all logic will be much simplified

    // Validate Color clear
    auto report_color_hazard = [this, &skip, &loc, &info](const HazardResult& hazard, const VkClearAttachment& clear_attachment,
                                                          uint32_t clear_rect_index, const VkClearRect& clear_rect) {
        std::ostringstream ss;
        ss << string_VkImageAspectFlags(clear_attachment.aspectMask);
        ss << " aspect of color attachment " << clear_attachment.colorAttachment;
        ss << " (" << sync_state_.FormatHandle(info.attachment_view) << ")";
        if (current_renderpass_context_) {
            ss << " in subpass " << current_renderpass_context_->GetCurrentSubpass();
        }
        const std::string resource_description = ss.str();
        const LogObjectList objlist(cb_state_->Handle(), info.attachment_view.Handle());
        const auto error = error_messages_.ClearAttachmentError(hazard, *this, loc.function, resource_description,
                                                                clear_attachment.aspectMask, clear_rect_index, clear_rect);
        skip |= sync_state_.SyncError(hazard.Hazard(), objlist, loc, error);
    };
    if (aspects_to_clear & kColorAspects) {
        // [core validation check]: if COLOR_ASPECT is included then PLANE aspects are not allowed,
        // and if PLANE aspect is included then only one is allowed.
        assert(CountSetBits(aspects_to_clear) == 1);

        const AttachmentAccess attachment_access = GetAttachmentAccess(SyncOrdering::kColorAttachment);
        if (view_mask == 0) {
            HazardResult hazard = current_context_->DetectAttachmentHazard(
                *info.attachment_view.image_state, subresource_range, info.attachment_view.is_depth_sliced,
                SYNC_COLOR_ATTACHMENT_OUTPUT_COLOR_ATTACHMENT_WRITE, attachment_access);
            if (hazard.IsHazard()) {
                report_color_hazard(hazard, clear_attachment, clear_rect_index, clear_rect);
            }
        } else {
            const auto view_indices = GetSetBitIndices(view_mask);
            const VkImageSubresourceRange& attachment_subresource = info.attachment_view.normalized_subresource_range;
            for (uint32_t view_index : view_indices) {
                if (view_index < attachment_subresource.layerCount) {
                    VkImageSubresourceRange view_subresource = attachment_subresource;
                    view_subresource.baseArrayLayer += view_index;
                    view_subresource.layerCount = 1;

                    ImageRangeGen range_gen = sub_state.MakeImageRangeGen(view_subresource, info.attachment_view.is_depth_sliced);
                    HazardResult hazard = current_context_->DetectAttachmentHazard(
                        range_gen, SYNC_COLOR_ATTACHMENT_OUTPUT_COLOR_ATTACHMENT_WRITE, attachment_access);
                    if (hazard.IsHazard()) {
                        report_color_hazard(hazard, clear_attachment, clear_rect_index, clear_rect);
                    }
                }
            }
        }
    }

    // Validate Depth-Stencil clear
    auto report_depth_stencil_hazard = [this, &skip, &loc, &info](const HazardResult& hazard,
                                                                  const VkClearAttachment& clear_attachment,
                                                                  uint32_t clear_rect_index, const VkClearRect& clear_rect) {
        std::ostringstream ss;
        ss << string_VkImageAspectFlags(clear_attachment.aspectMask);
        ss << " aspect(s) of depth-stencil attachment (";
        ss << sync_state_.FormatHandle(info.attachment_view) << ")";
        if (current_renderpass_context_) {
            ss << " in subpass " << current_renderpass_context_->GetCurrentSubpass();
        }
        const std::string resource_description = ss.str();
        const LogObjectList objlist(cb_state_->Handle(), info.attachment_view.Handle());
        const auto error = error_messages_.ClearAttachmentError(hazard, *this, loc.function, resource_description,
                                                                clear_attachment.aspectMask, clear_rect_index, clear_rect);
        skip |= sync_state_.SyncError(hazard.Hazard(), objlist, loc, error);
    };
    if (aspects_to_clear & kDepthStencilAspects) {
        const AttachmentAccess attachment_access = GetAttachmentAccess(SyncOrdering::kDepthStencilAttachment);

        if (view_mask == 0) {
            // vkCmdClearAttachments depth/stencil writes are executed by the EARLY_FRAGMENT_TESTS_BIT and LATE_FRAGMENT_TESTS_BIT
            // stages. The implementation tracks the most recent access, which happens in the LATE_FRAGMENT_TESTS_BIT stage.
            HazardResult hazard = current_context_->DetectAttachmentHazard(
                *info.attachment_view.image_state, subresource_range, info.attachment_view.is_depth_sliced,
                SYNC_LATE_FRAGMENT_TESTS_DEPTH_STENCIL_ATTACHMENT_WRITE, attachment_access);
            if (hazard.IsHazard()) {
                report_depth_stencil_hazard(hazard, clear_attachment, clear_rect_index, clear_rect);
            }
        } else {
            const auto view_indices = GetSetBitIndices(view_mask);
            const VkImageSubresourceRange& attachment_subresource = info.attachment_view.normalized_subresource_range;
            for (uint32_t view_index : view_indices) {
                if (view_index < attachment_subresource.layerCount) {
                    VkImageSubresourceRange view_subresource = attachment_subresource;
                    view_subresource.baseArrayLayer += view_index;
                    view_subresource.layerCount = 1;

                    ImageRangeGen range_gen = sub_state.MakeImageRangeGen(view_subresource, info.attachment_view.is_depth_sliced);
                    HazardResult hazard = current_context_->DetectAttachmentHazard(
                        range_gen, SYNC_LATE_FRAGMENT_TESTS_DEPTH_STENCIL_ATTACHMENT_WRITE, attachment_access);
                    if (hazard.IsHazard()) {
                        report_depth_stencil_hazard(hazard, clear_attachment, clear_rect_index, clear_rect);
                    }
                }
            }
        }
    }
    return skip;
}

ResourceAccessCommand CommandBufferContext::MakeClearAttachmentAccessCommand(const VkClearAttachment& clear_attachment,
                                                                             uint32_t clear_rect_index,
                                                                             const VkClearRect& clear_rect) const {
    ResourceAccessCommand command;
    const auto optional_info = GetClearAttachmentInfo(clear_attachment, clear_rect.baseArrayLayer, clear_rect.layerCount);
    if (!optional_info) {
        return command;
    }
    const ClearAttachmentInfo& info = *optional_info;
    const VkImageAspectFlags aspects_to_clear = info.subresource_range.aspectMask;
    const bool color_clear = (aspects_to_clear & kColorAspects) != 0;
    const SyncAccessIndex access_index =
        color_clear ? SYNC_COLOR_ATTACHMENT_OUTPUT_COLOR_ATTACHMENT_WRITE : SYNC_LATE_FRAGMENT_TESTS_DEPTH_STENCIL_ATTACHMENT_WRITE;
    const SyncOrdering ordering = color_clear ? SyncOrdering::kColorAttachment : SyncOrdering::kDepthStencilAttachment;

    std::ostringstream ss;
    if (color_clear) {
        ss << string_VkImageAspectFlags(clear_attachment.aspectMask) << " aspect of color attachment "
           << clear_attachment.colorAttachment;
    } else {
        ss << string_VkImageAspectFlags(clear_attachment.aspectMask) << " aspect(s) of depth-stencil attachment";
    }
    ss << " (" << sync_state_.FormatHandle(info.attachment_view) << ")";
    if (current_renderpass_context_) {
        ss << " in subpass " << current_renderpass_context_->GetCurrentSubpass();
    }
    const std::string resource_description = ss.str();

    auto add_access = [&](const VkImageSubresourceRange& subresource_range) {
        ResourceAccessCommand::ImageRangeAccess access;
        access.image = info.attachment_view.image_state;
        access.access_index = access_index;
        access.subresource_range = subresource_range;
        access.is_depth_sliced = info.attachment_view.is_depth_sliced;
        access.attachment_access = GetAttachmentAccess(ordering);
        access.tag_handle = info.attachment_view.Handle();
        access.resource_description = resource_description;
        access.error_type = ResourceAccessCommand::ImageRangeAccess::ErrorType::kClearAttachment;
        access.clear_aspects = clear_attachment.aspectMask;
        access.clear_rect_index = clear_rect_index;
        access.clear_rect = clear_rect;
        command.accesses.emplace_back(std::move(access));
    };

    const uint32_t view_mask = GetViewMask();
    if (view_mask == 0) {
        add_access(info.subresource_range);
    } else {
        const VkImageSubresourceRange& attachment_subresource = info.attachment_view.normalized_subresource_range;
        for (uint32_t view_index : GetSetBitIndices(view_mask)) {
            if (view_index < attachment_subresource.layerCount) {
                VkImageSubresourceRange view_subresource = attachment_subresource;
                view_subresource.baseArrayLayer += view_index;
                view_subresource.layerCount = 1;
                add_access(view_subresource);
            }
        }
    }
    return command;
}

void CommandBufferContext::RecordClearAttachment(ResourceUsageTag tag, const VkClearAttachment& clear_attachment,
                                                 const VkClearRect& rect) {
    const auto optional_info = GetClearAttachmentInfo(clear_attachment, rect.baseArrayLayer, rect.layerCount);
    if (!optional_info) {
        return;
    }
    const ClearAttachmentInfo& info = *optional_info;
    const VkImageSubresourceRange subresource_range = info.subresource_range;
    const VkImageAspectFlags aspects_to_clear = subresource_range.aspectMask;
    const uint32_t view_mask = GetViewMask();
    const ImageSubState& sub_state = SubState(*info.attachment_view.image_state);

    auto update_access_state = [this, aspects_to_clear, tag](ImageRangeGen& range_gen) {
        if (aspects_to_clear & kColorAspects) {
            const AttachmentAccess attachment_access = GetAttachmentAccess(SyncOrdering::kColorAttachment);
            current_context_->UpdateAttachmentAccessState(range_gen, SYNC_COLOR_ATTACHMENT_OUTPUT_COLOR_ATTACHMENT_WRITE,
                                                          attachment_access, ResourceUsageTagEx{tag});
        } else {
            const AttachmentAccess attachment_access = GetAttachmentAccess(SyncOrdering::kDepthStencilAttachment);
            current_context_->UpdateAttachmentAccessState(range_gen, SYNC_LATE_FRAGMENT_TESTS_DEPTH_STENCIL_ATTACHMENT_WRITE,
                                                          attachment_access, ResourceUsageTagEx{tag});
        }
    };
    // NOTE: when we teach ImageRangeGen to work with view masks all logic will be much simplified
    if (view_mask == 0) {
        ImageRangeGen range_gen = sub_state.MakeImageRangeGen(subresource_range, false);
        update_access_state(range_gen);
    } else {
        const auto view_indices = GetSetBitIndices(view_mask);
        const VkImageSubresourceRange& attachment_subresource = info.attachment_view.normalized_subresource_range;
        for (uint32_t view_index : view_indices) {
            if (view_index < attachment_subresource.layerCount) {
                VkImageSubresourceRange view_subresource = attachment_subresource;
                view_subresource.baseArrayLayer += view_index;
                view_subresource.layerCount = 1;
                ImageRangeGen range_gen = sub_state.MakeImageRangeGen(view_subresource, false);
                update_access_state(range_gen);
            }
        }
    }
}

QueueId CommandBufferContext::GetQueueId() const { return kQueueIdInvalid; }

ResourceUsageTag CommandBufferContext::RecordBeginRenderPass(
    vvl::Func command, const vvl::RenderPass& rp_state, const VkRect2D& render_area,
    vvl::span<const std::shared_ptr<const vvl::ImageView>> attachment_views, bool apply_command) {
    // Create an access context the current renderpass.
    const auto barrier_tag = NextCommandTag(command, SubCommandType::kSubpassTransition, 0);
    AddCommandHandle(barrier_tag, rp_state.Handle());
    const auto load_tag = NextSubCommandTag(command, SubCommandType::kLoadOp, 0);
    render_pass_contexts_.emplace_back(std::make_unique<RenderPassAccessContext>(
        rp_state, render_area, environment_.queue_flags, attachment_views, cb_access_context_, current_render_pass_instance_id_));
    current_renderpass_context_ = render_pass_contexts_.back().get();
    if (apply_command) {
        current_renderpass_context_->RecordBeginRenderPass(barrier_tag, load_tag);
    }
    current_context_ = &current_renderpass_context_->CurrentContext();
    return barrier_tag;
}

ResourceUsageTag CommandBufferContext::RecordNextSubpass(vvl::Func command, bool apply_command) {
    // At this point current subpass value has not updated yet to the index of "next subpass"
    const uint32_t previous_subpass = current_renderpass_context_->GetCurrentSubpass();
    const uint32_t this_subpass = previous_subpass + 1;

    auto resolve_tag = NextCommandTag(command, SubCommandType::kResolveOp, previous_subpass);
    AddCommandHandle(resolve_tag, current_renderpass_context_->GetRenderPassState()->Handle());
    auto store_tag = NextSubCommandTag(command, SubCommandType::kStoreOp, previous_subpass);
    auto transition_tag = NextSubCommandTag(command, SubCommandType::kSubpassTransition, this_subpass);
    auto load_tag = NextSubCommandTag(command, SubCommandType::kLoadOp, this_subpass);

    if (apply_command) {
        current_renderpass_context_->RecordNextSubpass(resolve_tag, store_tag, transition_tag, load_tag);
    } else {
        current_renderpass_context_->AdvanceSubpass();
    }
    current_context_ = &current_renderpass_context_->CurrentContext();
    return resolve_tag;
}

ResourceUsageTag CommandBufferContext::RecordEndRenderPass(vvl::Func command, bool apply_command) {
    const uint32_t current_subpass = current_renderpass_context_->GetCurrentSubpass();

    auto store_tag = NextCommandTag(command, SubCommandType::kStoreOp, current_subpass);
    AddCommandHandle(store_tag, current_renderpass_context_->GetRenderPassState()->Handle());

    auto barrier_tag = NextSubCommandTag(command, SubCommandType::kSubpassTransition);

    if (apply_command) {
        current_renderpass_context_->RecordEndRenderPass(&cb_access_context_, store_tag, barrier_tag);
    }
    current_context_ = &cb_access_context_;
    current_renderpass_context_ = nullptr;
    current_render_pass_instance_id_++;
    return store_tag;
}

void CommandBufferContext::RecordDestroyEvent(vvl::Event* event_state) { events_context_.Destroy(event_state); }

void CommandBufferContext::RecordExecutedCommandBuffer(const CommandBufferContext& recorded_cb_context, bool apply_commands) {
    const AccessContext& recorded_context = recorded_cb_context.GetCbAccessContext();
    const ResourceUsageTag base_tag = GetTagCount();

    if (apply_commands) {
        if (sync_state_.syncval_settings.full_validation) {
            assert(recorded_cb_context.HasCompleteRecordedCommandStream());
        }
        if (recorded_cb_context.HasCompleteRecordedCommandStream()) {
            ApplyRecordedCommands(environment_, *current_context_, recorded_cb_context, base_tag);
        } else {
            // Preserve the legacy secondary-command-buffer path when command replay is disabled.
            for (const ReplayEntry& entry : recorded_cb_context.GetReplayEntries()) {
                const bool replay_action = GetReplayContextChange(entry.operation) == nullptr;
                if (replay_action) {
                    ApplyReplayAction(environment_, entry.operation, *current_context_, base_tag + entry.tag);
                }
            }
            ResolveExecutedCommandBuffer(recorded_context, base_tag);
        }
    }

    ImportRecordedAccessLog(recorded_cb_context);
    ImportRecordedCommands(recorded_cb_context, base_tag);
}

void CommandBufferContext::ResolveExecutedCommandBuffer(const AccessContext& recorded_context, ResourceUsageTag offset) {
    auto tag_offset = [offset](AccessState* access) { access->OffsetTag(offset); };
    current_context_->ResolveFromContext(tag_offset, recorded_context);
}

void CommandBufferContext::ImportRecordedAccessLog(const CommandBufferContext& recorded_context) {
    cbs_referenced_->emplace_back(recorded_context.GetCBStateShared());
    access_log_->insert(access_log_->end(), recorded_context.access_log_->cbegin(), recorded_context.access_log_->cend());

    // Adjust command indices for the log records added from recorded_context.
    const auto& recorded_label_commands = recorded_context.cb_state_->GetLabelCommands();
    const bool use_proxy = !proxy_label_commands_.empty();
    const auto& label_commands = use_proxy ? proxy_label_commands_ : cb_state_->GetLabelCommands();
    if (!label_commands.empty()) {
        assert(label_commands.size() >= recorded_label_commands.size());
        const uint32_t command_offset = static_cast<uint32_t>(label_commands.size() - recorded_label_commands.size());
        for (size_t i = 0; i < recorded_context.access_log_->size(); i++) {
            size_t index = (access_log_->size() - 1) - i;
            assert((*access_log_)[index].label_command_index != vvl::kNoIndex32);
            (*access_log_)[index].label_command_index += command_offset;
        }
    }
}

void CommandBufferContext::ImportRecordedCommands(const CommandBufferContext& recorded_context, ResourceUsageTag offset) {
    for (const RecordedCommandEntry& entry : recorded_context.recorded_commands_) {
        std::visit(
            [&](const auto& source_storage) {
                using StorageType = std::decay_t<decltype(source_storage)>;
                if constexpr (std::is_same_v<StorageType, BufferCopyCommand::Storage>) {
                    const BufferCopyCommand command = source_storage.MakeCommand(recorded_context.command_data_);
                    auto destination_storage =
                        command.MakeStorage(command_data_, source_storage.src_handle_index, source_storage.dst_handle_index);
                    recorded_commands_.emplace_back(offset + entry.tag, std::move(destination_storage));
                } else if constexpr (std::is_same_v<StorageType, BufferAccessCommand::Storage>) {
                    const BufferAccessCommand command = source_storage.MakeCommand(recorded_context.command_data_);
                    auto destination_storage = command.MakeStorage(command_data_, source_storage.handle_index);
                    recorded_commands_.emplace_back(offset + entry.tag, std::move(destination_storage));
                } else if constexpr (std::is_same_v<StorageType, NoOpCommand>) {
                    recorded_commands_.emplace_back(offset + entry.tag, source_storage);
                } else {
                    const auto& command = source_storage.MakeCommand(recorded_context.command_data_);
                    auto destination_storage = command.MakeStorage(command_data_);
                    recorded_commands_.emplace_back(offset + entry.tag, std::move(destination_storage));
                }
            },
            entry.command);
    }
}

ResourceUsageTag CommandBufferContext::NextCommandTag(vvl::Func command, SubCommandType subcommand, uint32_t subpass) {
    command_number_++;
    current_command_tag_ = access_log_->size();

    ResourceUsageRecord& record = access_log_->emplace_back(command, command_number_, subcommand, cb_state_, reset_count_, subpass);

    if (!cb_state_->GetLabelCommands().empty()) {
        record.label_command_index = static_cast<uint32_t>(cb_state_->GetLabelCommands().size() - 1);
    }
    CheckCommandTagDebugCheckpoint();
    return current_command_tag_;
}

ResourceUsageTag CommandBufferContext::NextSubCommandTag(vvl::Func command, SubCommandType subcommand, uint32_t subpass) {
    const ResourceUsageTag tag = access_log_->size();
    ResourceUsageRecord& record = access_log_->emplace_back(command, command_number_, subcommand, cb_state_, reset_count_, subpass);

    // By default copy handle range from the main command, but can be overwritten with AddSubcommandHandle.
    const auto& main_command_record = (*access_log_)[current_command_tag_];
    record.first_handle_index = main_command_record.first_handle_index;
    record.handle_count = main_command_record.handle_count;

    if (!cb_state_->GetLabelCommands().empty()) {
        record.label_command_index = static_cast<uint32_t>(cb_state_->GetLabelCommands().size() - 1);
    }
    return tag;
}

uint32_t CommandBufferContext::AddHandle(const VulkanTypedHandle& typed_handle, uint32_t index) {
    const uint32_t handle_index = static_cast<uint32_t>(handles_.size());
    handles_.emplace_back(HandleRecord(typed_handle, index));
    sync_state_.stats.AddHandleRecord();
    return handle_index;
}

ResourceUsageTagEx CommandBufferContext::AddCommandHandle(ResourceUsageTag tag, const VulkanTypedHandle& typed_handle) {
    return AddCommandHandleIndexed(tag, typed_handle, vvl::kNoIndex32);
}

ResourceUsageTagEx CommandBufferContext::AddCommandHandleIndexed(ResourceUsageTag tag, const VulkanTypedHandle& typed_handle,
                                                                 uint32_t index) {
    assert(tag < access_log_->size());
    const uint32_t handle_index = AddHandle(typed_handle, index);
    // TODO: the following range check is not needed. Test and remove.
    if (tag < access_log_->size()) {
        auto& record = (*access_log_)[tag];
        if (record.first_handle_index == vvl::kNoIndex32) {
            record.first_handle_index = handle_index;
            record.handle_count = 1;
        } else {
            // assert that command handles occupy continuous range
            assert(handle_index - record.first_handle_index == record.handle_count);
            record.handle_count++;
        }
    }
    return {tag, handle_index};
}

void CommandBufferContext::AddSubcommandHandleIndexed(ResourceUsageTag tag, const VulkanTypedHandle& typed_handle, uint32_t index) {
    assert(tag < access_log_->size());
    const uint32_t handle_index = AddHandle(typed_handle, index);
    // TODO: the following range check is not needed. Test and remove.
    if (tag < access_log_->size()) {
        auto& record = (*access_log_)[tag];
        const auto& main_command_record = (*access_log_)[current_command_tag_];
        if (record.first_handle_index == main_command_record.first_handle_index) {
            // override default behavior that subcommand references the same handles as the main command
            record.first_handle_index = handle_index;
            record.handle_count = 1;
        } else {
            // assert that command handles occupy continuous range
            assert(handle_index - record.first_handle_index == record.handle_count);
            record.handle_count++;
        }
    }
}

std::string CommandBufferContext::GetDebugRegionName(const ResourceUsageRecord& record) const {
    const bool use_proxy = !proxy_label_commands_.empty();
    const auto& label_commands = use_proxy ? proxy_label_commands_ : cb_state_->GetLabelCommands();
    return vvl::CommandBuffer::GetDebugRegionName(label_commands, record.label_command_index);
}

AttachmentAccess CommandBufferContext::GetAttachmentAccess(SyncOrdering ordering, AttachmentAccessType type) const {
    AttachmentAccess attachment_access;
    attachment_access.type = type;
    attachment_access.ordering = ordering;
    attachment_access.render_pass_instance_id = current_render_pass_instance_id_;
    attachment_access.subpass = current_renderpass_context_ ? current_renderpass_context_->GetCurrentSubpass() : vvl::kNoIndex32;
    return attachment_access;
}

uint32_t CommandBufferContext::GetViewMask() const {
    if (dynamic_rendering_info_) {
        return dynamic_rendering_info_->info.viewMask;
    } else if (current_renderpass_context_) {
        const auto& render_pass_ci = current_renderpass_context_->GetRenderPassState()->create_info;
        const uint32_t subpass = current_renderpass_context_->GetCurrentSubpass();
        return render_pass_ci.pSubpasses[subpass].viewMask;
    } else {
        assert(false && "GetViewMask musk be called only during render pass instance");
        return 0;
    }
}

// NOTE: debug location reporting feature works only for reproducible application sessions
// (it uses command number/reset count from the error message from the previous session).
// It's considered experimental and can be replaced with a better way to report syncval debug locations.
//
// Logs informational message when vulkan command stream reaches a specific location.
// The message can be intercepted by the reporting routines. For example, the message handler can trigger a breakpoint.
// The location can be specified through environment variables.
// VK_SYNCVAL_DEBUG_COMMAND_NUMBER: the command number
// VK_SYNCVAL_DEBUG_RESET_COUNT: (optional, default value is 1) command buffer reset count
// VK_SYNCVAL_DEBUG_CMDBUF_PATTERN: (optional, empty string by default) pattern to match command buffer debug name
void CommandBufferContext::CheckCommandTagDebugCheckpoint() {
    auto get_cmdbuf_name = [](const DebugReport& debug_report, uint64_t cmdbuf_handle) {
        std::unique_lock<std::mutex> lock(debug_report.debug_output_mutex);
        std::string object_name = debug_report.GetUtilsObjectNameNoLock(cmdbuf_handle);
        if (object_name.empty()) {
            object_name = debug_report.GetMarkerObjectNameNoLock(cmdbuf_handle);
        }
        text::ToLower(object_name);
        return object_name;
    };
    if (sync_state_.debug_command_number == command_number_ && sync_state_.debug_reset_count == reset_count_) {
        const auto cmdbuf_name = get_cmdbuf_name(*sync_state_.debug_report, cb_state_->Handle().handle);
        const auto& pattern = sync_state_.debug_cmdbuf_pattern;
        const bool cmdbuf_match = pattern.empty() || (cmdbuf_name.find(pattern) != std::string::npos);
        if (cmdbuf_match) {
            sync_state_.LogInfo("SYNCVAL_DEBUG_COMMAND", LogObjectList(), Location(access_log_->back().command),
                                "Command stream has reached command #%" PRIu32 " in command buffer %s with reset count #%" PRIu32,
                                sync_state_.debug_command_number, sync_state_.FormatHandle(cb_state_->Handle()).c_str(),
                                sync_state_.debug_reset_count);
        }
    }
}

void UpdateAccessMapStats(const AccessMap& access_map, AccessContextStats& stats);

void CommandBufferContext::UpdateStats(AccessStats& access_stats) const {
#if VVL_ENABLE_SYNCVAL_STATS != 0
    UpdateAccessMapStats(cb_access_context_.GetAccessMap(), access_stats.cb_access_stats);

    for (const auto& render_pass_context : render_pass_contexts_) {
        for (const AccessContext& subpass_access_context : render_pass_context->GetSubpassContexts()) {
            UpdateAccessMapStats(subpass_access_context.GetAccessMap(), access_stats.subpass_access_stats);
        }
    }
#endif
}

CommandBufferSubState::CommandBufferSubState(SyncValidator& dev, vvl::CommandBuffer& cb)
    : vvl::CommandBufferSubState(cb), cb_context(dev, &cb) {
    cb_context.SetSelfReference();
}

void CommandBufferSubState::End() {
    cb_context.GetCbAccessContext().Finalize();

    // For threads that are dedicated to recording command buffers but do not submit themselves,
    // the end of recording is a logical point to update memory stats
    cb_context.GetSyncState().stats.UpdateMemoryStats();
}

void CommandBufferSubState::Destroy() {
    cb_context.Destroy();  // must be first to clean up self references correctly.
}

void CommandBufferSubState::Reset(const Location& loc) { cb_context.Reset(); }

void CommandBufferSubState::NotifyInvalidate(const vvl::StateObject::NodeList& invalid_nodes, bool unlink) {
    for (auto& obj : invalid_nodes) {
        switch (obj->Type()) {
            case kVulkanObjectTypeEvent:
                cb_context.RecordDestroyEvent(static_cast<vvl::Event*>(obj.get()));
                break;
            default:
                break;
        }
    }
}

void CommandBufferSubState::RecordCopyBuffer(vvl::Buffer& src_buffer_state, vvl::Buffer& dst_buffer_state, uint32_t region_count,
                                             const VkBufferCopy* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    const auto src_tag_ex = cb_context.AddCommandHandle(tag, src_buffer_state.Handle());
    const auto dst_tag_ex = cb_context.AddCommandHandle(tag, dst_buffer_state.Handle());

    small_vector<BufferCopyRegion, 1> command_regions;
    command_regions.reserve(region_count);
    for (const VkBufferCopy& region : vvl::make_span(regions, region_count)) {
        command_regions.emplace_back(BufferCopyRegion{region.srcOffset, region.dstOffset, region.size});
    }
    const BufferCopyCommand command{src_buffer_state, dst_buffer_state, command_regions};

    if (cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled()) {
        command.Apply(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), src_tag_ex, dst_tag_ex);
    }
    cb_context.AddRecordedCommand(tag, command, src_tag_ex.handle_index, dst_tag_ex.handle_index);
}

void CommandBufferSubState::RecordCopyBuffer2(vvl::Buffer& src_buffer_state, vvl::Buffer& dst_buffer_state, uint32_t region_count,
                                              const VkBufferCopy2* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    const auto src_tag_ex = cb_context.AddCommandHandle(tag, src_buffer_state.Handle());
    const auto dst_tag_ex = cb_context.AddCommandHandle(tag, dst_buffer_state.Handle());

    small_vector<BufferCopyRegion, 1> command_regions;
    command_regions.reserve(region_count);
    for (const VkBufferCopy2& region : vvl::make_span(regions, region_count)) {
        command_regions.emplace_back(BufferCopyRegion{region.srcOffset, region.dstOffset, region.size});
    }
    const BufferCopyCommand command{src_buffer_state, dst_buffer_state, command_regions};

    if (cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled()) {
        command.Apply(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), src_tag_ex, dst_tag_ex);
    }
    cb_context.AddRecordedCommand(tag, command, src_tag_ex.handle_index, dst_tag_ex.handle_index);
}

void CommandBufferSubState::RecordCopyImage(vvl::Image& src_image_state, vvl::Image& dst_image_state,
                                            VkImageLayout src_image_layout, VkImageLayout dst_image_layout, uint32_t region_count,
                                            const VkImageCopy* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageCopyCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                             cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyImage2(vvl::Image& src_image_state, vvl::Image& dst_image_state,
                                             VkImageLayout src_image_layout, VkImageLayout dst_image_layout, uint32_t region_count,
                                             const VkImageCopy2* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageCopyCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                             cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyBufferToImage(vvl::Buffer& src_buffer_state, vvl::Image& dst_image_state, VkImageLayout,
                                                    uint32_t region_count, const VkBufferImageCopy* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeBufferToImageCopyCommand(cb_context.GetSyncState().Get<vvl::Buffer>(src_buffer_state.VkHandle()),
                                     cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyBufferToImage2(vvl::Buffer& src_buffer_state, vvl::Image& dst_image_state, VkImageLayout,
                                                     uint32_t region_count, const VkBufferImageCopy2* regions,
                                                     const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeBufferToImageCopyCommand(cb_context.GetSyncState().Get<vvl::Buffer>(src_buffer_state.VkHandle()),
                                     cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyImageToBuffer(vvl::Image& src_image_state, vvl::Buffer& dst_buffer_state,
                                                    VkImageLayout src_image_layout, uint32_t region_count,
                                                    const VkBufferImageCopy* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = MakeImageToBufferCopyCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                                                cb_context.GetSyncState().Get<vvl::Buffer>(dst_buffer_state.VkHandle()),
                                                region_count, regions);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyImageToBuffer2(vvl::Image& src_image_state, vvl::Buffer& dst_buffer_state,
                                                     VkImageLayout src_image_layout, uint32_t region_count,
                                                     const VkBufferImageCopy2* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = MakeImageToBufferCopyCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                                                cb_context.GetSyncState().Get<vvl::Buffer>(dst_buffer_state.VkHandle()),
                                                region_count, regions);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordBlitImage(vvl::Image& src_image_state, vvl::Image& dst_image_state,
                                            VkImageLayout src_image_layout, VkImageLayout dst_image_layout, uint32_t region_count,
                                            const VkImageBlit* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageBlitCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                             cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordBlitImage2(vvl::Image& src_image_state, vvl::Image& dst_image_state,
                                             VkImageLayout src_image_layout, VkImageLayout dst_image_layout, uint32_t region_count,
                                             const VkImageBlit2* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageBlitCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                             cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordResolveImage(vvl::Image& src_image_state, vvl::Image& dst_image_state, uint32_t region_count,
                                               const VkImageResolve* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageResolveCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                                cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordResolveImage2(vvl::Image& src_image_state, vvl::Image& dst_image_state, uint32_t region_count,
                                                const VkImageResolve2* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageResolveCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                                cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordClearColorImage(vvl::Image& image_state, VkImageLayout, const VkClearColorValue*,
                                                  uint32_t range_count, const VkImageSubresourceRange* ranges,
                                                  const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = MakeImageClearCommand(cb_context.GetSyncState().Get<vvl::Image>(image_state.VkHandle()), range_count, ranges);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordClearDepthStencilImage(vvl::Image& image_state, VkImageLayout, const VkClearDepthStencilValue*,
                                                         uint32_t range_count, const VkImageSubresourceRange* ranges,
                                                         const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = MakeImageClearCommand(cb_context.GetSyncState().Get<vvl::Image>(image_state.VkHandle()), range_count, ranges);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordClearAttachments(uint32_t attachment_count, const VkClearAttachment* pAttachments,
                                                   uint32_t rect_count, const VkClearRect* pRects, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    ResourceAccessCommand command;
    for (const auto& attachment : vvl::make_span(pAttachments, attachment_count)) {
        for (const auto [rect_index, rect] : vvl::enumerate(pRects, rect_count)) {
            command.Append(cb_context.MakeClearAttachmentAccessCommand(attachment, rect_index, rect));
        }
    }
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordFillBuffer(vvl::Buffer& buffer_state, VkDeviceSize offset, VkDeviceSize size,
                                             const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    const AccessRange range = MakeRange(buffer_state, offset, size);
    const BufferAccessCommand command =
        MakeBufferAccessCommand(buffer_state, SYNC_CLEAR_TRANSFER_WRITE, range, 0, {}, "dstBuffer ");
    const auto tag_ex = cb_context.AddCommandHandle(tag, buffer_state.Handle());
    if (cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled()) {
        command.Apply(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), tag_ex);
    }
    cb_context.AddRecordedCommand(tag, command, tag_ex.handle_index);
}

void CommandBufferSubState::RecordUpdateBuffer(vvl::Buffer& buffer_state, VkDeviceSize offset, VkDeviceSize size,
                                               const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    // VK_WHOLE_SIZE not allowed
    const AccessRange range = MakeRange(offset, size);
    const BufferAccessCommand command =
        MakeBufferAccessCommand(buffer_state, SYNC_CLEAR_TRANSFER_WRITE, range, 0, {}, "dstBuffer ");
    const auto tag_ex = cb_context.AddCommandHandle(tag, buffer_state.Handle());
    if (cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled()) {
        command.Apply(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), tag_ex);
    }
    cb_context.AddRecordedCommand(tag, command, tag_ex.handle_index);
}

void CommandBufferSubState::RecordDecodeVideo(vvl::VideoSession& vs_state, const VkVideoDecodeInfoKHR& decode_info,
                                              const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = cb_context.GetSyncState().MakeDecodeVideoAccessCommand(cb_context.GetCBState(), decode_info);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordEncodeVideo(vvl::VideoSession& vs_state, const VkVideoEncodeInfoKHR& encode_info,
                                              const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = cb_context.GetSyncState().MakeEncodeVideoAccessCommand(cb_context.GetCBState(), encode_info);
    command.Record(cb_context, tag, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyQueryPoolResults(vvl::QueryPool& pool_state, vvl::Buffer& dst_buffer_state,
                                                       uint32_t first_query, uint32_t query_count, VkDeviceSize dst_offset,
                                                       VkDeviceSize stride, VkQueryResultFlags flags, const Location& loc) {
    if (query_count == 0) {
        return;
    }
    const auto tag = cb_context.NextCommandTag(loc.function);
    const uint32_t query_size = (flags & VK_QUERY_RESULT_64_BIT) ? 8 : 4;
    const VkDeviceSize range_size = (query_count - 1) * stride + query_size;
    const AccessRange range = MakeRange(dst_offset, range_size);
    const BufferAccessCommand command = MakeBufferAccessCommand(dst_buffer_state, SYNC_COPY_TRANSFER_WRITE, range, 0,
                                                                pool_state.VkHandle(), "dstBuffer ");
    const auto tag_ex = cb_context.AddCommandHandle(tag, dst_buffer_state.Handle());
    if (cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled()) {
        command.Apply(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), tag_ex);
    }
    cb_context.AddRecordedCommand(tag, command, tag_ex.handle_index);

    // TODO:Track VkQueryPool
}

void CommandBufferSubState::RecordBeginRenderPass(const VkRenderPassBeginInfo& render_pass_begin,
                                                  const VkSubpassBeginInfo& subpass_begin_info, const Location& loc) {
    if (!base.IsPrimary()) {
        return;  // [core validation check]: only primary command buffer can begin render pass
    }

    const SyncValidator& validator = cb_context.GetSyncState();
    auto rp_state = validator.Get<vvl::RenderPass>(render_pass_begin.renderPass);
    if (!rp_state) {
        return;
    }

    std::vector<std::shared_ptr<const vvl::ImageView>> attachments;
    auto fb_state = validator.Get<vvl::Framebuffer>(render_pass_begin.framebuffer);
    if (fb_state) {
        attachments = validator.device_state->GetAttachmentViews(render_pass_begin, *fb_state);
    }

    RenderPassCommand command;
    command.type = RenderPassCommand::Type::kBegin;
    command.render_pass = rp_state;
    command.attachments = attachments;
    command.render_area = render_pass_begin.renderArea;
    command.render_pass_instance_id = cb_context.GetCurrentRenderPassInstanceId();
    command.command = loc.function;
    const ResourceUsageTag begin_tag =
        command.Record(cb_context, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());

    const RenderPassAccessContext* rp_context = cb_context.GetCurrentRenderPassContext();
    cb_context.AddReplayEntry(begin_tag, true, ReplayContextChange(std::move(rp_state), std::move(attachments), rp_context));
    cb_context.AddRecordedCommand(begin_tag, std::move(command));
    cb_context.AddRecordedCommand(begin_tag + 1, NoOpCommand{});
}

void CommandBufferSubState::RecordNextSubpass(const VkSubpassBeginInfo& subpass_begin_info,
                                              const VkSubpassEndInfo* subpass_end_info, const Location& loc) {
    if (!base.IsPrimary()) {
        return;  // [core validation check]: only primary command buffer can start next subpass
    }
    if (!cb_context.GetCurrentRenderPassContext()) {
        return;  // [core validation check]: begin render pass was not called
    }
    RenderPassCommand command;
    command.type = RenderPassCommand::Type::kNext;
    command.command = loc.function;
    const ResourceUsageTag resolve_tag =
        command.Record(cb_context, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());

    cb_context.AddReplayEntry(resolve_tag + 2, true, ReplayContextChange(ReplayContextChange::Type::kNextSubpass));
    cb_context.AddRecordedCommand(resolve_tag, std::move(command));
    cb_context.AddRecordedCommand(resolve_tag + 1, NoOpCommand{});
    cb_context.AddRecordedCommand(resolve_tag + 2, NoOpCommand{});
    cb_context.AddRecordedCommand(resolve_tag + 3, NoOpCommand{});
}

void CommandBufferSubState::RecordEndRenderPass(const VkSubpassEndInfo* subpass_end_info, const Location& loc) {
    if (!base.IsPrimary()) {
        return;  // [core validation check]: only primary command buffer can end render pass
    }
    if (!cb_context.GetCurrentRenderPassContext()) {
        return;  // [core validation check]: begin render pass was not called
    }
    RenderPassCommand command;
    command.type = RenderPassCommand::Type::kEnd;
    command.command = loc.function;
    const ResourceUsageTag store_tag =
        command.Record(cb_context, cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());

    cb_context.AddReplayEntry(store_tag + 1, true, ReplayContextChange(ReplayContextChange::Type::kEndRenderPass));
    cb_context.AddRecordedCommand(store_tag, std::move(command));
    cb_context.AddRecordedCommand(store_tag + 1, NoOpCommand{});
}

void CommandBufferSubState::RecordExecuteCommand(vvl::CommandBuffer& secondary_command_buffer, uint32_t cmd_index,
                                                 const Location& loc) {
    ResourceUsageTag cb_tag;
    if (cmd_index == 0) {
        cb_tag = cb_context.NextCommandTag(loc.function, SubCommandType::kIndex);
        cb_context.AddCommandHandleIndexed(cb_tag, secondary_command_buffer.Handle(), cmd_index);
    } else {
        cb_tag = cb_context.NextSubCommandTag(loc.function, SubCommandType::kIndex);
        cb_context.AddSubcommandHandleIndexed(cb_tag, secondary_command_buffer.Handle(), cmd_index);
    }
    cb_context.AddRecordedCommand(cb_tag, NoOpCommand{});
    cb_context.RecordExecutedCommandBuffer(GetCommandBufferContext(secondary_command_buffer),
                                           cb_context.GetSyncState().syncval_settings.IsRecordTimeValidationEnabled());
}

}  // namespace syncval
