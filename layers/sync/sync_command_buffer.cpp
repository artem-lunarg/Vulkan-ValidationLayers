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
    if (!sync_state_.syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    DynamicRenderingCommand command;
    CollectBeginRenderingAccesses(cmd_state.GetRenderingInfo(), error_obj.location, command);
    return command.Validate(*this, error_obj.location);
}

void CommandBufferContext::RecordBeginRendering(BeginRenderingCmdState& cmd_state, const Location& loc) {
    const auto tag = NextCommandTag(loc.function);
    DynamicRenderingCommand command;
    CollectBeginRenderingAccesses(cmd_state.GetRenderingInfo(), loc, command);
    command.Record(*this, tag, ApplyAccessesOnRecord());
    AddRecordedCommand(tag, std::move(command));
    dynamic_rendering_info_ = std::move(cmd_state.info);
}

bool CommandBufferContext::ValidateEndRendering(const ErrorObject& error_obj) const {
    if (!sync_state_.syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    DynamicRenderingCommand command;
    CollectEndRenderingAccesses(error_obj.location, command);
    return command.Validate(*this, error_obj.location);
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
    DynamicRenderingCommand command;
    CollectEndRenderingAccesses(record_obj.location, command);
    command.Record(*this, store_tag, ApplyAccessesOnRecord());
    AddRecordedCommand(store_tag, std::move(command));
    current_render_pass_instance_id_++;
    dynamic_rendering_info_.reset();
}

void CommandBufferContext::CollectBeginRenderingAccesses(const DynamicRenderingInfo& info, const Location& loc,
                                                         DynamicRenderingCommand& command) const {
    // Load operations do not happen when resuming
    if (info.info.flags & VK_RENDERING_RESUMING_BIT) {
        return;
    }
    for (size_t i = 0; i < info.attachments.size(); i++) {
        const auto& attachment = info.attachments[i];
        if (!attachment.IsValid()) {
            continue;  // VkRenderingAttachmentInfo::imageView is allowed to be VK_NULL_HANDLE
        }
        const SyncAccessIndex load_index = attachment.GetLoadUsage();
        if (load_index == SYNC_ACCESS_INDEX_NONE) {
            continue;
        }
        std::ostringstream ss;
        ss << vvl::String(vvl::Field::pRenderingInfo) << ".";
        ss << attachment.GetLocation(loc, uint32_t(i)).Fields();
        ss << " (" << sync_state_.FormatHandle(attachment.view->Handle());
        ss << ", loadOp " << string_VkAttachmentLoadOp(attachment.info.loadOp) << ")";

        DynamicRenderingCommand::Access access = {attachment.view,
                                                  attachment.GetRangeGen(info.info.viewMask),
                                                  load_index,
                                                  GetAttachmentAccess(attachment.GetOrdering(), AttachmentAccessType::LoadOp),
                                                  DynamicRenderingCommand::OpType::kLoad,
                                                  uint32_t(attachment.info.loadOp),
                                                  ss.str()};
        command.accesses.emplace_back(std::move(access));
    }
}

void CommandBufferContext::CollectEndRenderingAccesses(const Location& loc, DynamicRenderingCommand& command) const {
    // Only resolve and store if not suspending (as specified by BeginRendering)
    if (!dynamic_rendering_info_ || (dynamic_rendering_info_->info.flags & VK_RENDERING_SUSPENDING_BIT) != 0) {
        return;
    }
    const DynamicRenderingInfo& info = *dynamic_rendering_info_;

    for (uint32_t i = 0; i < (uint32_t)info.attachments.size(); i++) {
        const auto& attachment = info.attachments[i];
        if (!attachment.IsValid()) {
            continue;  // VkRenderingAttachmentInfo::imageView is allowed to be VK_NULL_HANDLE
        }

        auto attachment_description = [this, &loc, &attachment, i](const auto& view, std::ostringstream& ss) {
            ss << vvl::String(vvl::Field::pRenderingInfo) << ".";
            ss << attachment.GetLocation(loc, i).Fields();
            ss << " (" << sync_state_.FormatHandle(view->Handle());
        };

        // The logic about whether to resolve is embedded in the Attachment constructor
        if (attachment.resolve_gen) {
            const bool is_color = attachment.type == AttachmentType::kColor;
            const SyncOrdering resolve_order = is_color ? kColorResolveOrder : kDepthStencilResolveOrder;
            {
                std::ostringstream ss;
                attachment_description(attachment.view, ss);
                ss << ", resolveMode " << string_VkResolveModeFlagBits(attachment.info.resolveMode) << ")";
                DynamicRenderingCommand::Access access = {attachment.view,
                                                          attachment.GetRangeGen(info.info.viewMask),
                                                          kResolveRead,
                                                          GetAttachmentAccess(resolve_order, AttachmentAccessType::ResolveRead),
                                                          DynamicRenderingCommand::OpType::kResolveRead,
                                                          uint32_t(attachment.info.resolveMode),
                                                          ss.str()};
                command.accesses.emplace_back(std::move(access));
            }
            {
                std::ostringstream ss;
                attachment_description(attachment.resolve_view, ss);
                ss << ", resolveMode " << string_VkResolveModeFlagBits(attachment.info.resolveMode) << ")";
                DynamicRenderingCommand::Access access = {attachment.resolve_view,
                                                          *attachment.resolve_gen,
                                                          kResolveWrite,
                                                          GetAttachmentAccess(resolve_order, AttachmentAccessType::ResolveWrite),
                                                          DynamicRenderingCommand::OpType::kResolveWrite,
                                                          uint32_t(attachment.info.resolveMode),
                                                          ss.str()};
                command.accesses.emplace_back(std::move(access));
            }
        }

        const SyncAccessIndex store_index = attachment.GetStoreUsage();
        if (store_index != SYNC_ACCESS_INDEX_NONE) {
            std::ostringstream ss;
            attachment_description(attachment.view, ss);
            ss << ", storeOp " << string_VkAttachmentStoreOp(attachment.info.storeOp) << ")";
            DynamicRenderingCommand::Access access = {attachment.view,
                                                      attachment.GetRangeGen(info.info.viewMask),
                                                      store_index,
                                                      GetAttachmentAccess(kStoreOrder, AttachmentAccessType::StoreOp),
                                                      DynamicRenderingCommand::OpType::kStore,
                                                      uint32_t(attachment.info.storeOp),
                                                      ss.str()};
            command.accesses.emplace_back(std::move(access));
        }
    }
}

void CommandBufferContext::CollectDescriptorAccesses(
    VkPipelineBindPoint pipeline_bind_point, std::vector<DescriptorAccess>& accesses,
    std::shared_ptr<const vvl::Pipeline>& pipeline, std::vector<std::shared_ptr<const vvl::DescriptorSet>>& descriptor_sets) const {
    if (!sync_state_.syncval_settings.shader_accesses_heuristic) {
        return;
    }

    const auto& last_bound_state = cb_state_->lastBound[ConvertToVvlBindPoint(pipeline_bind_point)];
    const vvl::Pipeline* pipe = last_bound_state.pipeline_state;
    const std::vector<LastBound::DescriptorSetSlot>& ds_slots = last_bound_state.ds_slots;
    if (!pipe) {
        return;
    }
    pipeline = std::static_pointer_cast<const vvl::Pipeline>(pipe->shared_from_this());

    auto get_set_index = [&descriptor_sets](const std::shared_ptr<vvl::DescriptorSet>& descriptor_set) {
        for (uint32_t i = 0; i < descriptor_sets.size(); i++) {
            if (descriptor_sets[i].get() == descriptor_set.get()) {
                return i;
            }
        }
        descriptor_sets.emplace_back(descriptor_set);
        return static_cast<uint32_t>(descriptor_sets.size() - 1);
    };

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
                // This should be caught by Core validation, but if core checks are disabled SyncVal should not crash.
                continue;
            }
            const auto& ds_slot = ds_slots[variable.decorations.set];
            const auto* descriptor_set = ds_slot.ds_state.get();
            if (!descriptor_set) continue;
            auto binding = descriptor_set->GetBinding(variable.decorations.binding);
            if (!binding) continue;
            const auto descriptor_type = binding->type;
            SyncAccessIndex sync_index = GetSyncStageAccessIndexsByDescriptorSet(descriptor_type, variable, stage_state.GetStage());

            // Do not collect accesses for descriptor arrays (matches the previous Validate/Record behavior)
            if (binding->count > 1) {
                continue;
            }

            for (uint32_t i = 0; i < binding->count; i++) {
                const auto* descriptor = binding->GetDescriptor(i);
                DescriptorAccess access;
                access.access_index = sync_index;
                access.stage = stage_state.GetStage();
                access.descriptor_type = descriptor_type;
                access.set_number = variable.decorations.set;
                access.binding = variable.decorations.binding;

                switch (descriptor->GetClass()) {
                    case DescriptorClass::ImageSampler:
                    case DescriptorClass::Image: {
                        // NOTE: ImageSamplerDescriptor inherits from ImageDescriptor, so this cast works for both types.
                        const auto* image_descriptor = static_cast<const ImageDescriptor*>(descriptor);
                        if (image_descriptor->Invalid()) {
                            continue;
                        }
                        const auto* img_view_state = image_descriptor->GetImageViewState();
                        if (img_view_state->is_depth_sliced) {
                            // NOTE: 2D ImageViews of VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT Images are not allowed in
                            // Descriptors, unless VK_EXT_image_2d_view_of_3d is supported, which it isn't at the moment.
                            // See: VUID 00343
                            continue;
                        }
                        access.image_view = std::static_pointer_cast<const vvl::ImageView>(img_view_state->shared_from_this());
                        access.image_layout = image_descriptor->GetImageLayout();
                        access.command_handle = img_view_state->image_state->Handle();
                        access.description_handle = img_view_state->Handle();
                        if (sync_index == SYNC_FRAGMENT_SHADER_INPUT_ATTACHMENT_READ) {
                            access.input_attachment = true;
                            access.render_offset = CastTo3D(cb_state_->render_area.offset);
                            access.render_extent = CastTo3D(cb_state_->render_area.extent);
                            access.attachment_access = GetAttachmentAccess(SyncOrdering::kRaster);
                        }
                        break;
                    }
                    case DescriptorClass::TexelBuffer: {
                        const auto* texel_descriptor = static_cast<const TexelDescriptor*>(descriptor);
                        if (texel_descriptor->Invalid()) {
                            continue;
                        }
                        const auto* buf_view_state = texel_descriptor->GetBufferViewState();
                        access.buffer = buf_view_state->buffer_state;
                        access.range = MakeRange(*buf_view_state);
                        access.command_handle = buf_view_state->Handle();
                        access.description_handle = buf_view_state->Handle();
                        break;
                    }
                    case DescriptorClass::GeneralBuffer: {
                        const auto* buffer_descriptor = static_cast<const BufferDescriptor*>(descriptor);
                        if (buffer_descriptor->Invalid()) {
                            continue;
                        }
                        VkDeviceSize offset = buffer_descriptor->GetOffset();
                        if (vvl::IsDynamicDescriptor(descriptor_type)) {
                            const uint32_t dynamic_offset_index =
                                descriptor_set->GetDynamicOffsetIndexFromBinding(binding->binding);
                            if (dynamic_offset_index >= ds_slot.dynamic_offsets.size()) {
                                continue;  // core validation error
                            }
                            offset += ds_slot.dynamic_offsets[dynamic_offset_index];
                        }
                        const auto* buf_state = buffer_descriptor->GetBufferState();
                        access.buffer =
                            std::static_pointer_cast<const vvl::Buffer>(const_cast<vvl::Buffer*>(buf_state)->shared_from_this());
                        access.range = MakeRange(*buf_state, offset, buffer_descriptor->GetRange());
                        access.command_handle = buf_state->Handle();
                        access.description_handle = buf_state->Handle();
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
                        access.buffer = std::static_pointer_cast<const vvl::Buffer>(
                            const_cast<vvl::Buffer*>(as_buffer.state)->shared_from_this());
                        access.range = MakeRange(*as_buffer.state, as_buffer.offset, accel->GetSize());
                        access.command_handle = accel->Handle();
                        access.description_handle = accel->Handle();
                        break;
                    }
                    // TODO: INLINE_UNIFORM_BLOCK_EXT
                    default:
                        continue;
                }
                access.set_index = get_set_index(ds_slot.ds_state);
                accesses.emplace_back(std::move(access));
            }
        }
    }
}

void CommandBufferContext::CollectDrawVertexAccesses(uint32_t vertex_count, uint32_t first_vertex, DrawCommand& command) const {
    const auto* pipe = cb_state_->GetLastBoundGraphics().pipeline_state;
    if (!pipe) {
        return;
    }
    const auto& binding_buffers = cb_state_->current_vertex_buffer_binding_info;
    const auto& vertex_bindings = pipe->IsDynamic(CB_DYNAMIC_STATE_VERTEX_INPUT_EXT)
                                      ? cb_state_->dynamic_state_value.vertex_bindings
                                      : pipe->vertex_input_state->bindings;

    for (const auto& [_, binding_state] : vertex_bindings) {
        const auto& binding_desc = binding_state.desc;
        if (binding_desc.inputRate != VK_VERTEX_INPUT_RATE_VERTEX) {
            // TODO: add support to determine range of instance level attributes
            continue;
        }
        if (const auto* vertex_buffer = vvl::Find(binding_buffers, binding_desc.binding)) {
            // TODO - Handle https://gitlab.khronos.org/vulkan/Vulkan-ValidationLayers/-/issues/45
            const auto buf_state = sync_state_.Get<vvl::Buffer>(vertex_buffer->Buffer());
            if (!buf_state) continue;  // also skips if using nullDescriptor

            const AccessRange range =
                MakeRangeForVertexData(vertex_buffer->BufferOffset(), first_vertex, vertex_count, binding_state);
            command.vertex_accesses.emplace_back(DrawCommand::VertexAccess{buf_state, range});
        }
    }
}

void CommandBufferContext::CollectDrawIndexAccess(uint32_t index_count, uint32_t first_index, DrawCommand& command) const {
    const auto& index_binding = cb_state_->index_buffer_binding;
    // TODO - Handle https://gitlab.khronos.org/vulkan/Vulkan-ValidationLayers/-/issues/45
    const auto index_buf_state = sync_state_.Get<vvl::Buffer>(index_binding.Buffer());
    if (!index_buf_state) {
        return;
    }
    const uint32_t index_size = IndexTypeByteSize(index_binding.index_type);
    const AccessRange range = MakeRangeForIndexData(index_binding.BufferOffset(), first_index, index_count, index_size);
    command.index_accesses.emplace_back(DrawCommand::VertexAccess{index_buf_state, range});

    // TODO: Shader instrumentation support is needed to read index buffer content and determine
    // the range of accessed vertices. Until then vertex accesses of indexed draws are not collected.
}

void CommandBufferContext::CollectDrawAttachmentAccesses(std::vector<DrawAttachmentAccess>& accesses) const {
    const auto& last_bound_state = cb_state_->GetLastBoundGraphics();
    const auto* pipe = last_bound_state.pipeline_state;
    if (!pipe || pipe->RasterizationDisabled()) {
        return;
    }
    if (current_renderpass_context_) {
        current_renderpass_context_->CollectDrawAttachmentAccesses(*cb_state_, accesses);
        return;
    }
    if (!dynamic_rendering_info_) {
        return;
    }
    const DynamicRenderingInfo& info = *dynamic_rendering_info_;
    const auto& list = pipe->fs_writable_output_location_list;

    for (const auto output_location : list) {
        if (output_location >= info.info.colorAttachmentCount) {
            continue;
        }
        const auto& attachment = info.attachments[output_location];
        if (!attachment.IsWriteable(last_bound_state)) {
            continue;
        }
        DrawAttachmentAccess access;
        access.view = attachment.view;
        access.range_gen = attachment.GetRangeGen(info.info.viewMask);
        access.access_index = SYNC_COLOR_ATTACHMENT_OUTPUT_COLOR_ATTACHMENT_WRITE;
        access.attachment_access = GetAttachmentAccess(SyncOrdering::kColorAttachment);
        access.dynamic_rendering = true;
        access.attachment_field = vvl::Field::pColorAttachments;
        access.attachment_index = output_location;
        accesses.emplace_back(std::move(access));
    }

    const uint32_t attachment_count = static_cast<uint32_t>(info.attachments.size());
    for (uint32_t i = info.info.colorAttachmentCount; i < attachment_count; i++) {
        const auto& attachment = info.attachments[i];
        if (!attachment.IsWriteable(last_bound_state)) {
            continue;
        }
        DrawAttachmentAccess access;
        access.view = attachment.view;
        access.range_gen = attachment.GetRangeGen(info.info.viewMask);
        access.access_index = SYNC_LATE_FRAGMENT_TESTS_DEPTH_STENCIL_ATTACHMENT_WRITE;
        access.attachment_access = GetAttachmentAccess(SyncOrdering::kDepthStencilAttachment);
        access.dynamic_rendering = true;
        access.attachment_field =
            attachment.type == AttachmentType::kDepth ? vvl::Field::pDepthAttachment : vvl::Field::pStencilAttachment;
        accesses.emplace_back(std::move(access));
    }
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

void CommandBufferContext::CollectClearAttachmentAccesses(const VkClearAttachment& clear_attachment, uint32_t rect_index,
                                                          const VkClearRect& clear_rect, ClearAttachmentsCommand& command) const {
    const auto optional_info = GetClearAttachmentInfo(clear_attachment, clear_rect.baseArrayLayer, clear_rect.layerCount);
    if (!optional_info) {
        return;
    }
    const ClearAttachmentInfo& info = *optional_info;
    const VkImageSubresourceRange subresource_range = info.subresource_range;
    const VkImageAspectFlags aspects_to_clear = subresource_range.aspectMask;
    const uint32_t view_mask = GetViewMask();
    const ImageSubState& sub_state = SubState(*info.attachment_view.image_state);

    SyncAccessIndex access_index;
    AttachmentAccess attachment_access;
    std::ostringstream ss;
    ss << string_VkImageAspectFlags(clear_attachment.aspectMask);
    if (aspects_to_clear & kColorAspects) {
        // [core validation check]: if COLOR_ASPECT is included then PLANE aspects are not allowed,
        // and if PLANE aspect is included then only one is allowed.
        assert(CountSetBits(aspects_to_clear) == 1);
        // vkCmdClearAttachments color writes are executed by the COLOR_ATTACHMENT_OUTPUT stage.
        access_index = SYNC_COLOR_ATTACHMENT_OUTPUT_COLOR_ATTACHMENT_WRITE;
        attachment_access = GetAttachmentAccess(SyncOrdering::kColorAttachment);
        ss << " aspect of color attachment " << clear_attachment.colorAttachment;
        ss << " (" << sync_state_.FormatHandle(info.attachment_view) << ")";
    } else {
        // vkCmdClearAttachments depth/stencil writes are executed by the EARLY_FRAGMENT_TESTS_BIT and LATE_FRAGMENT_TESTS_BIT
        // stages. The implementation tracks the most recent access, which happens in the LATE_FRAGMENT_TESTS_BIT stage.
        access_index = SYNC_LATE_FRAGMENT_TESTS_DEPTH_STENCIL_ATTACHMENT_WRITE;
        attachment_access = GetAttachmentAccess(SyncOrdering::kDepthStencilAttachment);
        ss << " aspect(s) of depth-stencil attachment (";
        ss << sync_state_.FormatHandle(info.attachment_view) << ")";
    }
    if (current_renderpass_context_) {
        ss << " in subpass " << current_renderpass_context_->GetCurrentSubpass();
    }
    const std::string description = ss.str();

    const auto view = std::static_pointer_cast<const vvl::ImageView>(info.attachment_view.shared_from_this());
    auto add_access = [&](ImageRangeGen&& range_gen) {
        command.accesses.emplace_back(ClearAttachmentsCommand::ClearAccess{view, std::move(range_gen), access_index,
                                                                           attachment_access, clear_attachment.aspectMask,
                                                                           rect_index, clear_rect, description});
    };

    // NOTE: when we teach ImageRangeGen to work with view masks all logic will be much simplified
    if (view_mask == 0) {
        add_access(sub_state.MakeImageRangeGen(subresource_range, info.attachment_view.is_depth_sliced));
    } else {
        const auto view_indices = GetSetBitIndices(view_mask);
        const VkImageSubresourceRange& attachment_subresource = info.attachment_view.normalized_subresource_range;
        for (uint32_t view_index : view_indices) {
            if (view_index < attachment_subresource.layerCount) {
                VkImageSubresourceRange view_subresource = attachment_subresource;
                view_subresource.baseArrayLayer += view_index;
                view_subresource.layerCount = 1;
                add_access(sub_state.MakeImageRangeGen(view_subresource, info.attachment_view.is_depth_sliced));
            }
        }
    }
}

QueueId CommandBufferContext::GetQueueId() const { return kQueueIdInvalid; }

ResourceUsageTag CommandBufferContext::RecordBeginRenderPass(
    vvl::Func command, const vvl::RenderPass& rp_state, const VkRect2D& render_area,
    const std::vector<std::shared_ptr<const vvl::ImageView>>& attachment_views) {
    // Create an access context the current renderpass.
    const auto barrier_tag = NextCommandTag(command, SubCommandType::kSubpassTransition, 0);
    AddCommandHandle(barrier_tag, rp_state.Handle());
    const auto load_tag = NextSubCommandTag(command, SubCommandType::kLoadOp, 0);
    render_pass_contexts_.emplace_back(std::make_unique<RenderPassAccessContext>(
        rp_state, render_area, environment_.queue_flags, attachment_views, cb_access_context_, current_render_pass_instance_id_));
    current_renderpass_context_ = render_pass_contexts_.back().get();
    if (ApplyAccessesOnRecord()) {
        current_renderpass_context_->RecordBeginRenderPass(barrier_tag, load_tag);
    }
    current_context_ = &current_renderpass_context_->CurrentContext();

    BeginRenderPassCommand begin_command;
    begin_command.rp_state = std::static_pointer_cast<const vvl::RenderPass>(rp_state.shared_from_this());
    begin_command.render_area = render_area;
    begin_command.attachment_views = attachment_views;
    begin_command.render_pass_instance_id = current_render_pass_instance_id_;
    AddRecordedCommand(barrier_tag, std::move(begin_command), 2);
    return barrier_tag;
}

ResourceUsageTag CommandBufferContext::RecordNextSubpass(vvl::Func command) {
    // At this point current subpass value has not updated yet to the index of "next subpass"
    const uint32_t previous_subpass = current_renderpass_context_->GetCurrentSubpass();
    const uint32_t this_subpass = previous_subpass + 1;

    auto resolve_tag = NextCommandTag(command, SubCommandType::kResolveOp, previous_subpass);
    AddCommandHandle(resolve_tag, current_renderpass_context_->GetRenderPassState()->Handle());
    auto store_tag = NextSubCommandTag(command, SubCommandType::kStoreOp, previous_subpass);
    auto transition_tag = NextSubCommandTag(command, SubCommandType::kSubpassTransition, this_subpass);
    auto load_tag = NextSubCommandTag(command, SubCommandType::kLoadOp, this_subpass);

    if (ApplyAccessesOnRecord()) {
        current_renderpass_context_->RecordNextSubpass(resolve_tag, store_tag, transition_tag, load_tag);
    } else {
        current_renderpass_context_->AdvanceSubpass();
    }
    current_context_ = &current_renderpass_context_->CurrentContext();
    AddRecordedCommand(resolve_tag, NextSubpassCommand{}, 4);
    return transition_tag;
}

ResourceUsageTag CommandBufferContext::RecordEndRenderPass(vvl::Func command) {
    const uint32_t current_subpass = current_renderpass_context_->GetCurrentSubpass();

    auto store_tag = NextCommandTag(command, SubCommandType::kStoreOp, current_subpass);
    AddCommandHandle(store_tag, current_renderpass_context_->GetRenderPassState()->Handle());

    auto barrier_tag = NextSubCommandTag(command, SubCommandType::kSubpassTransition);

    if (ApplyAccessesOnRecord()) {
        current_renderpass_context_->RecordEndRenderPass(&cb_access_context_, store_tag, barrier_tag);
    }
    current_context_ = &cb_access_context_;
    current_renderpass_context_ = nullptr;
    current_render_pass_instance_id_++;
    AddRecordedCommand(store_tag, EndRenderPassCommand{}, 2);
    return barrier_tag;
}

void CommandBufferContext::RecordDestroyEvent(vvl::Event* event_state) { events_context_.Destroy(event_state); }

bool CommandBufferContext::ApplyAccessesOnRecord() const { return sync_state_.syncval_settings.IsRecordTimeValidationEnabled(); }

bool CommandBufferContext::HasCompleteRecordedCommandStream() const {
    ResourceUsageTag next_tag = 0;
    for (const RecordedCommandEntry& entry : recorded_commands_) {
        if (entry.tag != next_tag) {
            return false;
        }
        if (const auto* execute_commands = std::get_if<ExecuteCommandsCommand>(&entry.command)) {
            const CommandBufferContext& secondary_context = GetCommandBufferContext(*execute_commands->secondary_cb);
            if (!secondary_context.HasCompleteRecordedCommandStream()) {
                return false;
            }
            // A secondary that was re-recorded after being executed (invalid usage) no longer
            // matches the imported log; fall back rather than replaying a diverged stream.
            if (secondary_context.GetTagCount() + 1 != entry.tag_count) {
                return false;
            }
        }
        next_tag += entry.tag_count;
    }
    return next_tag == access_log_->size();
}

void CommandBufferContext::RecordExecutedCommandBuffer(const CommandBufferContext& recorded_cb_context) {
    const ResourceUsageTag base_tag = GetTagCount();

    if (ApplyAccessesOnRecord()) {
        // Replay synchronization actions against the current destination state. The
        // secondary's recorded access state already includes their effects and is resolved below.
        for (const ReplayEntry& entry : recorded_cb_context.GetReplayEntries()) {
            const bool replay_action = GetReplayContextChange(entry.operation) == nullptr;
            if (replay_action) {
                ApplyReplayAction(environment_, entry.operation, *current_context_, base_tag + entry.tag);
            }
        }
        ResolveExecutedCommandBuffer(recorded_cb_context.GetCbAccessContext(), base_tag);
    }

    ImportRecordedAccessLog(recorded_cb_context);
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
    auto command =
        MakeBufferCopyCommand(cb_context.GetSyncState().Get<vvl::Buffer>(src_buffer_state.VkHandle()),
                              cb_context.GetSyncState().Get<vvl::Buffer>(dst_buffer_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyBuffer2(vvl::Buffer& src_buffer_state, vvl::Buffer& dst_buffer_state, uint32_t region_count,
                                              const VkBufferCopy2* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeBufferCopyCommand(cb_context.GetSyncState().Get<vvl::Buffer>(src_buffer_state.VkHandle()),
                              cb_context.GetSyncState().Get<vvl::Buffer>(dst_buffer_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyImage(vvl::Image& src_image_state, vvl::Image& dst_image_state,
                                            VkImageLayout src_image_layout, VkImageLayout dst_image_layout, uint32_t region_count,
                                            const VkImageCopy* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageCopyCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                             cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyImage2(vvl::Image& src_image_state, vvl::Image& dst_image_state,
                                             VkImageLayout src_image_layout, VkImageLayout dst_image_layout, uint32_t region_count,
                                             const VkImageCopy2* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageCopyCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                             cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyBufferToImage(vvl::Buffer& src_buffer_state, vvl::Image& dst_image_state, VkImageLayout,
                                                    uint32_t region_count, const VkBufferImageCopy* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = MakeBufferImageCopyCommand(cb_context.GetSyncState().Get<vvl::Buffer>(src_buffer_state.VkHandle()),
                                              cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), true,
                                              region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyBufferToImage2(vvl::Buffer& src_buffer_state, vvl::Image& dst_image_state, VkImageLayout,
                                                     uint32_t region_count, const VkBufferImageCopy2* regions,
                                                     const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = MakeBufferImageCopyCommand(cb_context.GetSyncState().Get<vvl::Buffer>(src_buffer_state.VkHandle()),
                                              cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), true,
                                              region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyImageToBuffer(vvl::Image& src_image_state, vvl::Buffer& dst_buffer_state,
                                                    VkImageLayout src_image_layout, uint32_t region_count,
                                                    const VkBufferImageCopy* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = MakeBufferImageCopyCommand(cb_context.GetSyncState().Get<vvl::Buffer>(dst_buffer_state.VkHandle()),
                                              cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()), false,
                                              region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordCopyImageToBuffer2(vvl::Image& src_image_state, vvl::Buffer& dst_buffer_state,
                                                     VkImageLayout src_image_layout, uint32_t region_count,
                                                     const VkBufferImageCopy2* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = MakeBufferImageCopyCommand(cb_context.GetSyncState().Get<vvl::Buffer>(dst_buffer_state.VkHandle()),
                                              cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()), false,
                                              region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordBlitImage(vvl::Image& src_image_state, vvl::Image& dst_image_state,
                                            VkImageLayout src_image_layout, VkImageLayout dst_image_layout, uint32_t region_count,
                                            const VkImageBlit* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageBlitCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                             cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordBlitImage2(vvl::Image& src_image_state, vvl::Image& dst_image_state,
                                             VkImageLayout src_image_layout, VkImageLayout dst_image_layout, uint32_t region_count,
                                             const VkImageBlit2* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageBlitCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                             cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordResolveImage(vvl::Image& src_image_state, vvl::Image& dst_image_state, uint32_t region_count,
                                               const VkImageResolve* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageResolveCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                                cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordResolveImage2(vvl::Image& src_image_state, vvl::Image& dst_image_state, uint32_t region_count,
                                                const VkImageResolve2* regions, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command =
        MakeImageResolveCommand(cb_context.GetSyncState().Get<vvl::Image>(src_image_state.VkHandle()),
                                cb_context.GetSyncState().Get<vvl::Image>(dst_image_state.VkHandle()), region_count, regions);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordClearColorImage(vvl::Image& image_state, VkImageLayout, const VkClearColorValue*,
                                                  uint32_t range_count, const VkImageSubresourceRange* ranges,
                                                  const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = MakeImageClearCommand(cb_context.GetSyncState().Get<vvl::Image>(image_state.VkHandle()), range_count, ranges);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordClearDepthStencilImage(vvl::Image& image_state, VkImageLayout, const VkClearDepthStencilValue*,
                                                         uint32_t range_count, const VkImageSubresourceRange* ranges,
                                                         const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    auto command = MakeImageClearCommand(cb_context.GetSyncState().Get<vvl::Image>(image_state.VkHandle()), range_count, ranges);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordClearAttachments(uint32_t attachment_count, const VkClearAttachment* pAttachments,
                                                   uint32_t rect_count, const VkClearRect* pRects, const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    ClearAttachmentsCommand command;
    for (const auto& attachment : vvl::make_span(pAttachments, attachment_count)) {
        for (const auto [rect_index, rect] : vvl::enumerate(pRects, rect_count)) {
            cb_context.CollectClearAttachmentAccesses(attachment, static_cast<uint32_t>(rect_index), rect, command);
        }
    }
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordFillBuffer(vvl::Buffer& buffer_state, VkDeviceSize offset, VkDeviceSize size,
                                             const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    const AccessRange range = MakeRange(buffer_state, offset, size);
    auto command = MakeBufferAccessCommand(cb_context.GetSyncState().Get<vvl::Buffer>(buffer_state.VkHandle()),
                                           SYNC_CLEAR_TRANSFER_WRITE, range);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordUpdateBuffer(vvl::Buffer& buffer_state, VkDeviceSize offset, VkDeviceSize size,
                                               const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    // VK_WHOLE_SIZE not allowed
    const AccessRange range = MakeRange(offset, size);
    auto command = MakeBufferAccessCommand(cb_context.GetSyncState().Get<vvl::Buffer>(buffer_state.VkHandle()),
                                           SYNC_CLEAR_TRANSFER_WRITE, range);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordDecodeVideo(vvl::VideoSession& vs_state, const VkVideoDecodeInfoKHR& decode_info,
                                              const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    VideoCommand command;
    cb_context.GetSyncState().AddVideoDecodeAccesses(command, vs_state, decode_info);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));
}

void CommandBufferSubState::RecordEncodeVideo(vvl::VideoSession& vs_state, const VkVideoEncodeInfoKHR& encode_info,
                                              const Location& loc) {
    const auto tag = cb_context.NextCommandTag(loc.function);
    VideoCommand command;
    cb_context.GetSyncState().AddVideoEncodeAccesses(command, vs_state, encode_info);
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
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
    auto command = MakeBufferAccessCommand(cb_context.GetSyncState().Get<vvl::Buffer>(dst_buffer_state.VkHandle()),
                                           SYNC_COPY_TRANSFER_WRITE, range, 0, pool_state.Handle());
    command.Record(cb_context, tag, cb_context.ApplyAccessesOnRecord());
    cb_context.AddRecordedCommand(tag, std::move(command));

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

    const ResourceUsageTag begin_tag =
        cb_context.RecordBeginRenderPass(loc.function, *rp_state, render_pass_begin.renderArea, attachments);
    const RenderPassAccessContext* rp_context = cb_context.GetCurrentRenderPassContext();
    cb_context.AddReplayEntry(begin_tag, true, ReplayContextChange(std::move(rp_state), std::move(attachments), rp_context));
}

void CommandBufferSubState::RecordNextSubpass(const VkSubpassBeginInfo& subpass_begin_info,
                                              const VkSubpassEndInfo* subpass_end_info, const Location& loc) {
    if (!base.IsPrimary()) {
        return;  // [core validation check]: only primary command buffer can start next subpass
    }
    if (!cb_context.GetCurrentRenderPassContext()) {
        return;  // [core validation check]: begin render pass was not called
    }
    const ResourceUsageTag tag = cb_context.RecordNextSubpass(loc.function);
    cb_context.AddReplayEntry(tag, true, ReplayContextChange(ReplayContextChange::Type::kNextSubpass));
}

void CommandBufferSubState::RecordEndRenderPass(const VkSubpassEndInfo* subpass_end_info, const Location& loc) {
    if (!base.IsPrimary()) {
        return;  // [core validation check]: only primary command buffer can end render pass
    }
    if (!cb_context.GetCurrentRenderPassContext()) {
        return;  // [core validation check]: begin render pass was not called
    }
    const ResourceUsageTag tag = cb_context.RecordEndRenderPass(loc.function);
    cb_context.AddReplayEntry(tag, true, ReplayContextChange(ReplayContextChange::Type::kEndRenderPass));
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
    const CommandBufferContext& secondary_context = GetCommandBufferContext(secondary_command_buffer);
    cb_context.RecordExecutedCommandBuffer(secondary_context);
    const uint32_t tag_count = static_cast<uint32_t>(cb_context.GetTagCount() - cb_tag);
    cb_context.AddRecordedCommand(cb_tag, ExecuteCommandsCommand{secondary_context.GetCBStateShared()}, tag_count);
}

}  // namespace syncval
