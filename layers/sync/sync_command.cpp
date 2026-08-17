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
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sync/sync_command.h"

#include "error_message/error_location.h"
#include "state_tracker/buffer_state.h"
#include "state_tracker/descriptor_sets.h"
#include "state_tracker/image_state.h"
#include "state_tracker/pipeline_state.h"
#include "state_tracker/ray_tracing_state.h"
#include "sync/sync_command_buffer.h"
#include "sync/sync_event.h"
#include "sync/sync_image.h"
#include "sync/sync_validation.h"
#include "utils/image_utils.h"

#include <optional>
#include <type_traits>

namespace syncval {
namespace {

}  // namespace

// Reports a hazard found while replaying recorded commands: at queue submission, or when
// vkCmdExecuteCommands validates an executed secondary command buffer against the primary.
// Reproduces the legacy first-use report: message shape, object list and resource description.
bool ReportReplayHazard(const SyncEnvironment& env, const CommandBufferContext& cb_context, const HazardResult& hazard,
                        ResourceUsageTag replay_tag, const VulkanTypedHandle& resource_handle, const Location& loc) {
    const SyncValidator& sync_state = env.validator;
    const LogObjectList objlist(env.handle, cb_context.GetCBState().Handle());
    const std::string resource_description =
        (resource_handle != NullVulkanTypedHandle) ? sync_state.FormatHandle(resource_handle) : "resource";
    const std::string error =
        sync_state.error_messages_.SubmitTimeError(env, hazard, cb_context, replay_tag, loc.index, resource_description);
    return sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
}

namespace {

LogObjectList CommandObjectList(const SyncEnvironment& env, const CommandBufferContext& cb_context,
                                const VulkanTypedHandle& resource, VkQueryPool query_pool = VK_NULL_HANDLE) {
    LogObjectList objects(env.handle);
    if (env.queue_id != kQueueIdInvalid) {
        objects.add(cb_context.GetCBState().Handle());
    }
    if (query_pool != VK_NULL_HANDLE) {
        objects.add(VulkanTypedHandle(query_pool, kVulkanObjectTypeQueryPool));
    }
    objects.add(resource);
    return objects;
}

}  // namespace

uint32_t CommandData::AddBuffer(const vvl::Buffer& buffer) {
    const uint32_t index = uint32_t(buffers.size());
    buffers.emplace_back(std::static_pointer_cast<const vvl::Buffer>(buffer.shared_from_this()));
    return index;
}

ResourceAccessCommand ResourceAccessCommand::Storage::MakeCommand(const CommandData& command_data) const {
    vvl::span<const Access> accesses;
    if (access_count != 0) {
        accesses = vvl::make_span(&command_data.resource_accesses[first_access], access_count);
    }
    return {CommandList<Access>(accesses)};
}

ResourceAccessCommand::Storage ResourceAccessCommand::MakeStorage(CommandData& command_data) const& {
    const uint32_t first_access = uint32_t(command_data.resource_accesses.size());
    const uint32_t access_count = uint32_t(accesses.size());
    accesses.AppendTo(command_data.resource_accesses);
    return {first_access, access_count};
}

ResourceAccessCommand::Storage ResourceAccessCommand::MakeStorage(CommandData& command_data) && {
    const uint32_t first_access = uint32_t(command_data.resource_accesses.size());
    const uint32_t access_count = uint32_t(accesses.size());
    std::move(accesses).AppendTo(command_data.resource_accesses);
    return {first_access, access_count};
}

ImageTransferCommand ImageTransferCommand::Storage::MakeCommand(const CommandData& command_data) const {
    vvl::span<const Access> accesses;
    if (access_count != 0) {
        accesses = vvl::make_span(&command_data.image_transfer_accesses[first_access], access_count);
    }
    return {CommandList<Access>(accesses)};
}

ImageTransferCommand::Storage ImageTransferCommand::MakeStorage(CommandData& command_data) const& {
    const uint32_t first_access = uint32_t(command_data.image_transfer_accesses.size());
    const uint32_t access_count = uint32_t(accesses.size());
    accesses.AppendTo(command_data.image_transfer_accesses);
    return {first_access, access_count};
}

ImageTransferCommand::Storage ImageTransferCommand::MakeStorage(CommandData& command_data) && {
    const uint32_t first_access = uint32_t(command_data.image_transfer_accesses.size());
    const uint32_t access_count = uint32_t(accesses.size());
    std::move(accesses).AppendTo(command_data.image_transfer_accesses);
    return {first_access, access_count};
}

const PipelineBarrierCommand& PipelineBarrierCommand::Storage::MakeCommand(const CommandData& command_data) const {
    return command_data.pipeline_barrier_commands[command_index];
}

PipelineBarrierCommand::Storage PipelineBarrierCommand::MakeStorage(CommandData& command_data) const& {
    const uint32_t index = uint32_t(command_data.pipeline_barrier_commands.size());
    command_data.pipeline_barrier_commands.emplace_back(*this);
    return {index};
}

PipelineBarrierCommand::Storage PipelineBarrierCommand::MakeStorage(CommandData& command_data) && {
    const uint32_t index = uint32_t(command_data.pipeline_barrier_commands.size());
    command_data.pipeline_barrier_commands.emplace_back(std::move(*this));
    return {index};
}

EventCommand EventCommand::Storage::MakeCommand(const CommandData& command_data) const {
    vvl::span<const std::shared_ptr<const vvl::Event>> events;
    if (event_count != 0) {
        events = vvl::make_span(&command_data.events[first_event], event_count);
    }
    vvl::span<const BarrierSet> barrier_sets;
    if (barrier_set_count != 0) {
        barrier_sets = vvl::make_span(&command_data.event_barrier_sets[first_barrier_set], barrier_set_count);
    }
    return {type, CommandList<std::shared_ptr<const vvl::Event>>(events), exec_scope, CommandList<BarrierSet>(barrier_sets), command};
}

EventCommand::Storage EventCommand::MakeStorage(CommandData& command_data) const& {
    const uint32_t first_event = uint32_t(command_data.events.size());
    const uint32_t event_count = uint32_t(events.size());
    events.AppendTo(command_data.events);
    const uint32_t first_barrier_set = uint32_t(command_data.event_barrier_sets.size());
    const uint32_t barrier_set_count = uint32_t(barrier_sets.size());
    barrier_sets.AppendTo(command_data.event_barrier_sets);
    return {type, first_event, event_count, exec_scope, first_barrier_set, barrier_set_count, command};
}

EventCommand::Storage EventCommand::MakeStorage(CommandData& command_data) && {
    const uint32_t first_event = uint32_t(command_data.events.size());
    const uint32_t event_count = uint32_t(events.size());
    std::move(events).AppendTo(command_data.events);
    const uint32_t first_barrier_set = uint32_t(command_data.event_barrier_sets.size());
    const uint32_t barrier_set_count = uint32_t(barrier_sets.size());
    std::move(barrier_sets).AppendTo(command_data.event_barrier_sets);
    return {type, first_event, event_count, exec_scope, first_barrier_set, barrier_set_count, command};
}

RenderPassCommand RenderPassCommand::Storage::MakeCommand(const CommandData& command_data) const {
    std::shared_ptr<const vvl::RenderPass> render_pass;
    if (render_pass_index != vvl::kNoIndex32) {
        render_pass = command_data.render_passes[render_pass_index];
    }
    vvl::span<const std::shared_ptr<const vvl::ImageView>> attachments;
    if (attachment_count != 0) {
        attachments = vvl::make_span(&command_data.render_pass_attachments[first_attachment], attachment_count);
    }
    return {type, std::move(render_pass), CommandList<std::shared_ptr<const vvl::ImageView>>(attachments), render_area,
            render_pass_instance_id, command};
}

RenderPassCommand::Storage RenderPassCommand::MakeStorage(CommandData& command_data) const& {
    uint32_t render_pass_index = vvl::kNoIndex32;
    if (render_pass) {
        render_pass_index = uint32_t(command_data.render_passes.size());
        command_data.render_passes.emplace_back(render_pass);
    }
    const uint32_t first_attachment = uint32_t(command_data.render_pass_attachments.size());
    const uint32_t attachment_count = uint32_t(attachments.size());
    attachments.AppendTo(command_data.render_pass_attachments);
    return {type, render_pass_index, first_attachment, attachment_count, render_area, render_pass_instance_id, command};
}

RenderPassCommand::Storage RenderPassCommand::MakeStorage(CommandData& command_data) && {
    uint32_t render_pass_index = vvl::kNoIndex32;
    if (render_pass) {
        render_pass_index = uint32_t(command_data.render_passes.size());
        command_data.render_passes.emplace_back(std::move(render_pass));
    }
    const uint32_t first_attachment = uint32_t(command_data.render_pass_attachments.size());
    const uint32_t attachment_count = uint32_t(attachments.size());
    std::move(attachments).AppendTo(command_data.render_pass_attachments);
    return {type, render_pass_index, first_attachment, attachment_count, render_area, render_pass_instance_id, command};
}

BufferCopyCommand BufferCopyCommand::Storage::MakeCommand(const CommandData& command_data) const {
    const vvl::Buffer& src_buffer = *command_data.buffers[src_buffer_index];
    const vvl::Buffer& dst_buffer = *command_data.buffers[dst_buffer_index];
    vvl::span<const BufferCopyRegion> regions;
    if (region_count != 0) {
        regions = vvl::make_span(&command_data.buffer_copy_regions[first_region], region_count);
    }
    return {src_buffer, dst_buffer, regions};
}

BufferCopyCommand::Storage BufferCopyCommand::MakeStorage(CommandData& command_data, uint32_t src_handle_index,
                                                          uint32_t dst_handle_index) const {
    const uint32_t src_buffer_index = command_data.AddBuffer(src_buffer);
    const uint32_t dst_buffer_index = command_data.AddBuffer(dst_buffer);

    const uint32_t first_region = uint32_t(command_data.buffer_copy_regions.size());
    const uint32_t region_count = uint32_t(regions.size());
    command_data.buffer_copy_regions.insert(command_data.buffer_copy_regions.end(), regions.begin(), regions.end());

    return {src_buffer_index, dst_buffer_index, first_region, region_count, src_handle_index, dst_handle_index};
}

BufferAccessCommand MakeBufferAccessCommand(const vvl::Buffer& buffer, SyncAccessIndex access_index, AccessRange range,
                                            uint8_t flags, VkQueryPool query_pool, const char* resource_name) {
    return BufferAccessCommand{buffer, access_index, range, flags, query_pool, resource_name};
}

BufferAccessCommand BufferAccessCommand::Storage::MakeCommand(const CommandData& command_data) const {
    return {*command_data.buffers[buffer_index], access_index, range, flags, query_pool, resource_name};
}

BufferAccessCommand::Storage BufferAccessCommand::MakeStorage(CommandData& command_data, uint32_t handle_index) const {
    return {command_data.AddBuffer(buffer), access_index, range, flags, query_pool, resource_name, handle_index};
}

bool BufferAccessCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    const bool is_marker = (flags & SyncFlag::kMarker) != 0;
    const AccessContext& access_context = is_marker ? cb_context.GetCurrentAccessContext() : cb_context.GetCbAccessContext();
    return Validate(cb_context.GetSyncEnvironment(), access_context, cb_context, loc.function, ResourceUsageRange{}, loc);
}

bool BufferAccessCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                   const CommandBufferContext& cb_context, vvl::Func command,
                                   const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                                   ResourceUsageTag replay_tag) const {
    const bool is_marker = (flags & SyncFlag::kMarker) != 0;
    const auto hazard =
        is_marker ? access_context.DetectMarkerHazard(buffer, range) : access_context.DetectHazard(buffer, access_index, range);
    if (!hazard.IsHazard() || record_time_validated_tags.includes(hazard.Tag())) {
        return false;
    }
    if (replay_tag != kInvalidTag) {
        return ReportReplayHazard(env, cb_context, hazard, replay_tag, buffer.Handle(), loc);
    }

    const SyncValidator& sync_state = env.validator;
    const LogObjectList objlist = is_marker ? LogObjectList(buffer.Handle())
                                            : CommandObjectList(env, cb_context, buffer.Handle(), query_pool);
    const std::string resource_description = resource_name + sync_state.FormatHandle(buffer.Handle());
    const std::string error = sync_state.error_messages_.BufferError(hazard, env, command, resource_description, range);
    return sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
}

void BufferAccessCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTagEx tag_ex) const {
    access_context.UpdateAccessState(buffer, access_index, range, tag_ex, flags, env.queue_id);
}

bool ResourceAccessCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool ResourceAccessCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                     const CommandBufferContext& cb_context, vvl::Func command,
                                     const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                                     ResourceUsageTag replay_tag) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    for (const Access& access : accesses) {
        std::visit(
            [&](const auto& value) {
                using AccessType = std::decay_t<decltype(value)>;
                HazardResult hazard;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    if (!value.buffer) {
                        return;
                    }
                    hazard = (value.flags & SyncFlag::kMarker)
                                 ? access_context.DetectMarkerHazard(*value.buffer, value.range)
                                 : access_context.DetectHazard(*value.buffer, value.access_index, value.range);
                } else if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    if (!value.image_view) {
                        return;
                    }
                    if (value.use_render_area || value.view_mask != 0) {
                        ImageRangeGen range_gen;
                        if (value.view_mask != 0) {
                            range_gen = MakeImageRangeGen(*value.image_view, value.view_mask, value.aspect_mask);
                        } else {
                            range_gen = MakeImageRangeGen(*value.image_view, value.offset, value.extent, value.aspect_mask);
                        }
                        hazard = access_context.DetectAttachmentHazard(range_gen, value.access_index, value.attachment_access,
                                                                       env.queue_id);
                    } else {
                        hazard = access_context.DetectHazard(*value.image_view, value.access_index);
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageRangeAccess>) {
                    if (!value.image) {
                        return;
                    }
                    const auto& sub_state = SubState(*value.image);
                    ImageRangeGen range_gen = value.use_offset_extent
                                                  ? sub_state.MakeImageRangeGen(value.subresource_range, value.offset, value.extent,
                                                                                value.is_depth_sliced)
                                                  : sub_state.MakeImageRangeGen(value.subresource_range, value.is_depth_sliced);
                    if (value.attachment_access.type == AttachmentAccessType::Empty) {
                        hazard = access_context.DetectHazard(range_gen, value.access_index);
                    } else {
                        hazard = access_context.DetectAttachmentHazard(range_gen, value.access_index, value.attachment_access,
                                                                       env.queue_id);
                    }
                }

                if (!hazard.IsHazard() || record_time_validated_tags.includes(hazard.Tag())) {
                    return;
                }
                if (replay_tag != kInvalidTag) {
                    // The legacy report derives the resource from the recorded command handle (kNoIndex32
                    // means the access did not register a handle and reports a generic "resource").
                    const VulkanTypedHandle resource_handle =
                        (value.handle_index != vvl::kNoIndex32) ? value.tag_handle : NullVulkanTypedHandle;
                    skip |= ReportReplayHazard(env, cb_context, hazard, replay_tag, resource_handle, loc);
                    return;
                }

                const DescriptorInfo* descriptor_info = nullptr;
                if constexpr (std::is_same_v<AccessType, BufferAccess> || std::is_same_v<AccessType, ImageViewAccess>) {
                    descriptor_info = value.descriptor_info ? &*value.descriptor_info : nullptr;
                }

                VulkanTypedHandle resource_handle;
                VulkanTypedHandle object_handle;
                if (descriptor_info) {
                    resource_handle = descriptor_info->resource_handle;
                    if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                        object_handle = descriptor_info->resource_type == DescriptorResourceType::kAccelerationStructure
                                            ? value.buffer->Handle()
                                            : resource_handle;
                    } else {
                        object_handle = resource_handle;
                    }
                } else if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    resource_handle = value.buffer->Handle();
                    object_handle = resource_handle;
                } else if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    resource_handle = value.image_view->Handle();
                    object_handle = resource_handle;
                } else {
                    resource_handle = value.tag_handle;
                    object_handle = resource_handle;
                }

                LogObjectList objlist;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    if (value.legacy_record_time_object_only) {
                        objlist = LogObjectList(object_handle);
                    } else {
                        objlist = CommandObjectList(env, cb_context, object_handle);
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageRangeAccess>) {
                    if (value.error_type == ImageRangeAccess::ErrorType::kVideo) {
                        objlist = LogObjectList(object_handle);
                    } else {
                        objlist = CommandObjectList(env, cb_context, object_handle);
                    }
                } else {
                    objlist = CommandObjectList(env, cb_context, object_handle);
                }
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    if (value.pipeline) {
                        objlist.add(value.pipeline->Handle());
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    if (value.additional_object != NullVulkanTypedHandle) {
                        objlist.add(value.additional_object);
                    }
                }
                std::string error;
                if (descriptor_info && descriptor_info->pipeline && descriptor_info->descriptor_set) {
                    objlist.add(descriptor_info->pipeline->Handle());
                    const std::string resource_description = sync_state.FormatHandle(resource_handle);
                    switch (descriptor_info->resource_type) {
                        case DescriptorResourceType::kBuffer:
                            error = sync_state.error_messages_.BufferDescriptorError(
                                hazard, env, command, resource_description, *descriptor_info->pipeline, descriptor_info->set,
                                *descriptor_info->descriptor_set, descriptor_info->descriptor_type, descriptor_info->binding,
                                descriptor_info->array_element, descriptor_info->stage);
                            break;
                        case DescriptorResourceType::kImage:
                            error = sync_state.error_messages_.ImageDescriptorError(
                                hazard, env, command, resource_description, *descriptor_info->pipeline, descriptor_info->set,
                                *descriptor_info->descriptor_set, descriptor_info->descriptor_type, descriptor_info->binding,
                                descriptor_info->array_element, descriptor_info->stage, descriptor_info->image_layout);
                            break;
                        case DescriptorResourceType::kAccelerationStructure:
                            error = sync_state.error_messages_.AccelerationStructureDescriptorError(
                                hazard, env, command, resource_description, *descriptor_info->pipeline, descriptor_info->set,
                                *descriptor_info->descriptor_set, descriptor_info->descriptor_type, descriptor_info->binding,
                                descriptor_info->array_element, descriptor_info->stage);
                            break;
                    }
                } else if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    const std::string resource_description = value.resource_name + sync_state.FormatHandle(resource_handle);
                    if (value.acceleration_structure_info && value.acceleration_structure_info->acceleration_structure) {
                        const auto& acceleration_structure = *value.acceleration_structure_info->acceleration_structure;
                        objlist.add(acceleration_structure.Handle());
                        error = sync_state.error_messages_.AccelerationStructureError(
                            hazard, env, command, resource_description, value.range, acceleration_structure.VkHandle(),
                            value.acceleration_structure_info->location);
                    } else {
                        error = sync_state.error_messages_.BufferError(hazard, env, command, resource_description, value.range);
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    std::string resource_description =
                        value.resource_description.empty() ? sync_state.FormatHandle(resource_handle) : value.resource_description;
                    if (!value.resource_description.empty() && value.additional_object != NullVulkanTypedHandle) {
                        resource_description += " (" + sync_state.FormatHandle(resource_handle) + ", " +
                                                sync_state.FormatHandle(value.additional_object) + ")";
                    }
                    error = sync_state.error_messages_.Error(env, hazard, command, resource_description, value.message_type);
                } else {
                    switch (value.error_type) {
                        case ImageRangeAccess::ErrorType::kClearAttachment:
                            error = sync_state.error_messages_.ClearAttachmentError(hazard, env, command,
                                                                                    value.resource_description, value.clear_aspects,
                                                                                    value.clear_rect_index, value.clear_rect);
                            break;
                        case ImageRangeAccess::ErrorType::kVideo:
                            error = sync_state.error_messages_.VideoError(hazard, env, command, value.resource_description);
                            break;
                        case ImageRangeAccess::ErrorType::kGeneric:
                            error = sync_state.error_messages_.Error(env, hazard, command, value.resource_description,
                                                                     "ImageRangeAccessError");
                            break;
                    }
                }
                if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    if (value.error_location != ImageViewAccess::ErrorLocation::kNone) {
                        auto report_at = [&](const Location& error_loc) {
                            skip |= sync_state.SyncError(hazard.Hazard(), objlist, error_loc, error);
                        };
                        switch (value.error_location) {
                            case ImageViewAccess::ErrorLocation::kColorAttachment: {
                                const Location attachment_loc = loc.dot(vvl::Struct::VkRenderingAttachmentInfo,
                                                                        vvl::Field::pColorAttachments, value.attachment_index);
                                const Location image_view_loc = attachment_loc.dot(vvl::Field::imageView);
                                report_at(image_view_loc);
                                break;
                            }
                            case ImageViewAccess::ErrorLocation::kDepthAttachment: {
                                const Location attachment_loc =
                                    loc.dot(vvl::Struct::VkRenderingAttachmentInfo, vvl::Field::pDepthAttachment);
                                const Location image_view_loc = attachment_loc.dot(vvl::Field::imageView);
                                report_at(image_view_loc);
                                break;
                            }
                            case ImageViewAccess::ErrorLocation::kStencilAttachment: {
                                const Location attachment_loc =
                                    loc.dot(vvl::Struct::VkRenderingAttachmentInfo, vvl::Field::pStencilAttachment);
                                const Location image_view_loc = attachment_loc.dot(vvl::Field::imageView);
                                report_at(image_view_loc);
                                break;
                            }
                            case ImageViewAccess::ErrorLocation::kNone:
                                break;
                        }
                    } else {
                        skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
                    }
                } else {
                    skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
                }
            },
            access);
    }
    return skip;
}

void ResourceAccessCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    for (Access& access : accesses.Mutable()) {
        std::visit(
            [&](auto& value) {
                if (value.tag_handle != NullVulkanTypedHandle) {
                    value.handle_index = cb_context.AddCommandHandle(tag, value.tag_handle).handle_index;
                }
            },
            access);
    }
    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), tag);
    }
}

void ResourceAccessCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    for (const Access& access : accesses) {
        std::visit(
            [&](const auto& value) {
                using AccessType = std::decay_t<decltype(value)>;
                const ResourceUsageTagEx tag_ex{tag, value.handle_index};
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    if (value.buffer && value.apply_access) {
                        access_context.UpdateAccessState(*value.buffer, value.access_index, value.range, tag_ex, value.flags,
                                                         env.queue_id);
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    if (!value.image_view) {
                        return;
                    }
                    if (value.use_render_area || value.view_mask != 0) {
                        ImageRangeGen range_gen;
                        if (value.view_mask != 0) {
                            range_gen = MakeImageRangeGen(*value.image_view, value.view_mask, value.aspect_mask);
                        } else {
                            range_gen = MakeImageRangeGen(*value.image_view, value.offset, value.extent, value.aspect_mask);
                        }
                        access_context.UpdateAttachmentAccessState(range_gen, value.access_index, value.attachment_access, tag_ex,
                                                                   env.queue_id);
                    } else {
                        ImageRangeGen range_gen = MakeImageRangeGen(*value.image_view);
                        access_context.UpdateAccessState(range_gen, value.access_index, tag_ex, 0, env.queue_id);
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageRangeAccess>) {
                    if (!value.image) {
                        return;
                    }
                    const auto& sub_state = SubState(*value.image);
                    ImageRangeGen range_gen = value.use_offset_extent
                                                  ? sub_state.MakeImageRangeGen(value.subresource_range, value.offset, value.extent,
                                                                                value.is_depth_sliced)
                                                  : sub_state.MakeImageRangeGen(value.subresource_range, value.is_depth_sliced);
                    if (value.attachment_access.type == AttachmentAccessType::Empty) {
                        access_context.UpdateAccessState(range_gen, value.access_index, tag_ex, 0, env.queue_id);
                    } else {
                        access_context.UpdateAttachmentAccessState(range_gen, value.access_index, value.attachment_access, tag_ex,
                                                                   env.queue_id);
                    }
                }
            },
            access);
    }
}

void ResourceAccessCommand::Append(ResourceAccessCommand&& other) {
    accesses.Append(std::move(other.accesses));
}

namespace {

template <typename RegionType>
ImageTransferCommand MakeImageCopyCommandImpl(std::shared_ptr<const vvl::Image> src_image,
                                              std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                              const RegionType* regions, SyncAccessIndex src_access, SyncAccessIndex dst_access) {
    ImageTransferCommand command;
    command.accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_image) {
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{src_image, src_access, region.srcSubresource,
                                                                            region.srcOffset, region.extent, region_index});
        }
        if (dst_image) {
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{dst_image, dst_access, region.dstSubresource,
                                                                            region.dstOffset, region.extent, region_index});
        }
    }
    return command;
}

template <typename RegionType>
ImageTransferCommand MakeBufferToImageCopyCommandImpl(std::shared_ptr<const vvl::Buffer> src_buffer,
                                                      std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                                      const RegionType* regions) {
    ImageTransferCommand command;
    command.accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_buffer && dst_image) {
            const AccessRange range = MakeRange(region.bufferOffset, dst_image->GetBufferSizeFromCopyImage(region));
            command.accesses.emplace_back(
                ImageTransferCommand::BufferAccess{src_buffer, SYNC_COPY_TRANSFER_READ, range, region_index});
        }
        if (dst_image) {
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{dst_image, SYNC_COPY_TRANSFER_WRITE,
                                                                            region.imageSubresource, region.imageOffset,
                                                                            region.imageExtent, region_index});
        }
    }
    return command;
}

template <typename RegionType>
ImageTransferCommand MakeImageToBufferCopyCommandImpl(std::shared_ptr<const vvl::Image> src_image,
                                                      std::shared_ptr<const vvl::Buffer> dst_buffer, uint32_t region_count,
                                                      const RegionType* regions) {
    ImageTransferCommand command;
    command.accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_image) {
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{
                src_image, SYNC_COPY_TRANSFER_READ, region.imageSubresource, region.imageOffset, region.imageExtent, region_index});
        }
        if (src_image && dst_buffer) {
            const AccessRange range = MakeRange(region.bufferOffset, src_image->GetBufferSizeFromCopyImage(region));
            command.accesses.emplace_back(
                ImageTransferCommand::BufferAccess{dst_buffer, SYNC_COPY_TRANSFER_WRITE, range, region_index});
        }
    }
    return command;
}

template <typename RegionType>
ImageTransferCommand MakeImageBlitCommandImpl(std::shared_ptr<const vvl::Image> src_image,
                                              std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                              const RegionType* regions) {
    ImageTransferCommand command;
    command.accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_image) {
            const VkOffset3D offset = {std::min(region.srcOffsets[0].x, region.srcOffsets[1].x),
                                       std::min(region.srcOffsets[0].y, region.srcOffsets[1].y),
                                       std::min(region.srcOffsets[0].z, region.srcOffsets[1].z)};
            const VkExtent3D extent = {static_cast<uint32_t>(std::abs(region.srcOffsets[1].x - region.srcOffsets[0].x)),
                                       static_cast<uint32_t>(std::abs(region.srcOffsets[1].y - region.srcOffsets[0].y)),
                                       static_cast<uint32_t>(std::abs(region.srcOffsets[1].z - region.srcOffsets[0].z))};
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{src_image, SYNC_BLIT_TRANSFER_READ,
                                                                            region.srcSubresource, offset, extent, region_index});
        }
        if (dst_image) {
            const VkOffset3D offset = {std::min(region.dstOffsets[0].x, region.dstOffsets[1].x),
                                       std::min(region.dstOffsets[0].y, region.dstOffsets[1].y),
                                       std::min(region.dstOffsets[0].z, region.dstOffsets[1].z)};
            const VkExtent3D extent = {static_cast<uint32_t>(std::abs(region.dstOffsets[1].x - region.dstOffsets[0].x)),
                                       static_cast<uint32_t>(std::abs(region.dstOffsets[1].y - region.dstOffsets[0].y)),
                                       static_cast<uint32_t>(std::abs(region.dstOffsets[1].z - region.dstOffsets[0].z))};
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{dst_image, SYNC_BLIT_TRANSFER_WRITE,
                                                                            region.dstSubresource, offset, extent, region_index});
        }
    }
    return command;
}

}  // namespace

ImageTransferCommand MakeImageCopyCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageCopy* regions) {
    return MakeImageCopyCommandImpl(std::move(src_image), std::move(dst_image), region_count, regions, SYNC_COPY_TRANSFER_READ,
                                    SYNC_COPY_TRANSFER_WRITE);
}

ImageTransferCommand MakeImageCopyCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageCopy2* regions) {
    return MakeImageCopyCommandImpl(std::move(src_image), std::move(dst_image), region_count, regions, SYNC_COPY_TRANSFER_READ,
                                    SYNC_COPY_TRANSFER_WRITE);
}

ImageTransferCommand MakeBufferToImageCopyCommand(std::shared_ptr<const vvl::Buffer> src_buffer,
                                                  std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                                  const VkBufferImageCopy* regions) {
    return MakeBufferToImageCopyCommandImpl(std::move(src_buffer), std::move(dst_image), region_count, regions);
}

ImageTransferCommand MakeBufferToImageCopyCommand(std::shared_ptr<const vvl::Buffer> src_buffer,
                                                  std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                                  const VkBufferImageCopy2* regions) {
    return MakeBufferToImageCopyCommandImpl(std::move(src_buffer), std::move(dst_image), region_count, regions);
}

ImageTransferCommand MakeImageToBufferCopyCommand(std::shared_ptr<const vvl::Image> src_image,
                                                  std::shared_ptr<const vvl::Buffer> dst_buffer, uint32_t region_count,
                                                  const VkBufferImageCopy* regions) {
    return MakeImageToBufferCopyCommandImpl(std::move(src_image), std::move(dst_buffer), region_count, regions);
}

ImageTransferCommand MakeImageToBufferCopyCommand(std::shared_ptr<const vvl::Image> src_image,
                                                  std::shared_ptr<const vvl::Buffer> dst_buffer, uint32_t region_count,
                                                  const VkBufferImageCopy2* regions) {
    return MakeImageToBufferCopyCommandImpl(std::move(src_image), std::move(dst_buffer), region_count, regions);
}

ImageTransferCommand MakeImageBlitCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageBlit* regions) {
    return MakeImageBlitCommandImpl(std::move(src_image), std::move(dst_image), region_count, regions);
}

ImageTransferCommand MakeImageBlitCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageBlit2* regions) {
    return MakeImageBlitCommandImpl(std::move(src_image), std::move(dst_image), region_count, regions);
}

ImageTransferCommand MakeImageResolveCommand(std::shared_ptr<const vvl::Image> src_image,
                                             std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                             const VkImageResolve* regions) {
    return MakeImageCopyCommandImpl(std::move(src_image), std::move(dst_image), region_count, regions, SYNC_RESOLVE_TRANSFER_READ,
                                    SYNC_RESOLVE_TRANSFER_WRITE);
}

ImageTransferCommand MakeImageResolveCommand(std::shared_ptr<const vvl::Image> src_image,
                                             std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                             const VkImageResolve2* regions) {
    return MakeImageCopyCommandImpl(std::move(src_image), std::move(dst_image), region_count, regions, SYNC_RESOLVE_TRANSFER_READ,
                                    SYNC_RESOLVE_TRANSFER_WRITE);
}

ImageTransferCommand MakeImageClearCommand(std::shared_ptr<const vvl::Image> image, uint32_t range_count,
                                           const VkImageSubresourceRange* ranges) {
    ImageTransferCommand command;
    command.accesses.reserve(range_count);
    if (image) {
        for (const auto [range_index, range] : vvl::enumerate(ranges, range_count)) {
            command.accesses.emplace_back(
                ImageTransferCommand::ImageRangeAccess{image, SYNC_CLEAR_TRANSFER_WRITE, range, range_index});
        }
    }
    return command;
}

bool ImageTransferCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool ImageTransferCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                    const CommandBufferContext& cb_context, vvl::Func command,
                                    const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                                    ResourceUsageTag replay_tag) const {
    bool skip = false;
    std::optional<uint32_t> hazard_region;
    for (const Access& access : accesses) {
        const uint32_t region_index = std::visit(
            [](const auto& value) {
                using AccessType = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<AccessType, ImageRangeAccess>) {
                    return value.range_index;
                } else {
                    return value.region_index;
                }
            },
            access);
        if (hazard_region && *hazard_region != region_index) {
            break;
        }

        std::visit(
            [&](const auto& value) {
                using AccessType = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    if (!value.buffer) {
                        return;
                    }
                    const auto hazard = access_context.DetectHazard(*value.buffer, value.access_index, value.range);
                    if (hazard.IsHazard() && !record_time_validated_tags.includes(hazard.Tag())) {
                        hazard_region = value.region_index;
                        if (replay_tag != kInvalidTag) {
                            skip |= ReportReplayHazard(env, cb_context, hazard, replay_tag, value.buffer->Handle(), loc);
                            return;
                        }
                        const LogObjectList objlist = CommandObjectList(env, cb_context, value.buffer->Handle());
                        const std::string error = env.validator.error_messages_.BufferCopyError(
                            hazard, env, command, env.validator.FormatHandle(value.buffer->Handle()), value.region_index,
                            value.range);
                        skip |= env.validator.SyncError(hazard.Hazard(), objlist, loc, error);
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageAccess>) {
                    if (!value.image) {
                        return;
                    }
                    const auto hazard = access_context.DetectHazard(*value.image, RangeFromLayers(value.subresource), value.offset,
                                                                    value.extent, value.access_index);
                    if (hazard.IsHazard() && !record_time_validated_tags.includes(hazard.Tag())) {
                        hazard_region = value.region_index;
                        if (replay_tag != kInvalidTag) {
                            skip |= ReportReplayHazard(env, cb_context, hazard, replay_tag, value.image->Handle(), loc);
                            return;
                        }
                        const LogObjectList objlist = CommandObjectList(env, cb_context, value.image->Handle());
                        const std::string error = env.validator.error_messages_.ImageCopyResolveBlitError(
                            hazard, env, command, env.validator.FormatHandle(value.image->Handle()), value.region_index,
                            value.offset, value.extent, value.subresource);
                        skip |= env.validator.SyncError(hazard.Hazard(), objlist, loc, error);
                    }
                } else {
                    if (!value.image) {
                        return;
                    }
                    const auto hazard = access_context.DetectHazard(*value.image, value.subresource_range, value.access_index);
                    if (hazard.IsHazard() && !record_time_validated_tags.includes(hazard.Tag())) {
                        hazard_region = value.range_index;
                        if (replay_tag != kInvalidTag) {
                            skip |= ReportReplayHazard(env, cb_context, hazard, replay_tag, value.image->Handle(), loc);
                            return;
                        }
                        const LogObjectList objlist = CommandObjectList(env, cb_context, value.image->Handle());
                        const std::string error = env.validator.error_messages_.ImageClearError(
                            hazard, env, command, env.validator.FormatHandle(value.image->Handle()), value.range_index,
                            value.subresource_range);
                        skip |= env.validator.SyncError(hazard.Hazard(), objlist, loc, error);
                    }
                }
            },
            access);
    }
    return skip;
}

void ImageTransferCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    for (Access& access : accesses.Mutable()) {
        std::visit(
            [&](auto& value) {
                using AccessType = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    if (value.buffer) {
                        value.handle_index = cb_context.AddCommandHandle(tag, value.buffer->Handle()).handle_index;
                    }
                } else {
                    if (value.image) {
                        value.handle_index = cb_context.AddCommandHandle(tag, value.image->Handle()).handle_index;
                    }
                }
            },
            access);
    }
    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), tag);
    }
}

void ImageTransferCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    for (const Access& access : accesses) {
        std::visit(
            [&](const auto& value) {
                using AccessType = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    if (value.buffer) {
                        access_context.UpdateAccessState(*value.buffer, value.access_index, value.range,
                                                         ResourceUsageTagEx{tag, value.handle_index}, 0, env.queue_id);
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageAccess>) {
                    if (!value.image) {
                        return;
                    }
                    ImageRangeGen range_gen =
                        SubState(*value.image)
                            .MakeImageRangeGen(RangeFromLayers(value.subresource), value.offset, value.extent, false);
                    access_context.UpdateAccessState(range_gen, value.access_index, ResourceUsageTagEx{tag, value.handle_index}, 0,
                                                     env.queue_id);
                } else if (value.image) {
                    ImageRangeGen range_gen = SubState(*value.image).MakeImageRangeGen(value.subresource_range, false);
                    access_context.UpdateAccessState(range_gen, value.access_index, ResourceUsageTagEx{tag, value.handle_index}, 0,
                                                     env.queue_id);
                }
            },
            access);
    }
}

bool BufferCopyCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool BufferCopyCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                 const CommandBufferContext& cb_context, vvl::Func command,
                                 const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                                 ResourceUsageTag replay_tag) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    for (uint32_t region_index = 0; region_index < regions.size(); ++region_index) {
        const BufferCopyRegion& region = regions[region_index];
        const AccessRange src_range = MakeRange(src_buffer, region.src_offset, region.size);
        const auto src_hazard = access_context.DetectHazard(src_buffer, SYNC_COPY_TRANSFER_READ, src_range);
        if (src_hazard.IsHazard() && !record_time_validated_tags.includes(src_hazard.Tag())) {
            if (replay_tag != kInvalidTag) {
                skip |= ReportReplayHazard(env, cb_context, src_hazard, replay_tag, src_buffer.Handle(), loc);
            } else {
                const LogObjectList objlist = CommandObjectList(env, cb_context, src_buffer.Handle());
                const std::string error = sync_state.error_messages_.BufferCopyError(
                    src_hazard, env, command, sync_state.FormatHandle(src_buffer.Handle()), region_index, src_range);
                skip |= sync_state.SyncError(src_hazard.Hazard(), objlist, loc, error);
            }
        }
        const AccessRange dst_range = MakeRange(dst_buffer, region.dst_offset, region.size);
        const auto dst_hazard = access_context.DetectHazard(dst_buffer, SYNC_COPY_TRANSFER_WRITE, dst_range);
        if (dst_hazard.IsHazard() && !record_time_validated_tags.includes(dst_hazard.Tag())) {
            if (replay_tag != kInvalidTag) {
                skip |= ReportReplayHazard(env, cb_context, dst_hazard, replay_tag, dst_buffer.Handle(), loc);
            } else {
                const LogObjectList objlist = CommandObjectList(env, cb_context, dst_buffer.Handle());
                const std::string error = sync_state.error_messages_.BufferCopyError(
                    dst_hazard, env, command, sync_state.FormatHandle(dst_buffer.Handle()), region_index, dst_range);
                skip |= sync_state.SyncError(dst_hazard.Hazard(), objlist, loc, error);
            }
        }
        if (skip) {
            break;
        }
    }
    return skip;
}

void BufferCopyCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTagEx src_tag_ex,
                              ResourceUsageTagEx dst_tag_ex) const {
    for (const BufferCopyRegion& region : regions) {
        const AccessRange src_range = MakeRange(src_buffer, region.src_offset, region.size);
        access_context.UpdateAccessState(src_buffer, SYNC_COPY_TRANSFER_READ, src_range, src_tag_ex, 0, env.queue_id);

        const AccessRange dst_range = MakeRange(dst_buffer, region.dst_offset, region.size);
        access_context.UpdateAccessState(dst_buffer, SYNC_COPY_TRANSFER_WRITE, dst_range, dst_tag_ex, 0, env.queue_id);
    }
}

bool PipelineBarrierCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool PipelineBarrierCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                      const CommandBufferContext& cb_context, vvl::Func command,
                                      const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                                      ResourceUsageTag replay_tag) const {
    bool skip = false;

    for (const auto& image_barrier : barriers.image_barriers) {
        if (!image_barrier.layout_transition) {
            continue;
        }
        const vvl::Image& image_state = *image_barrier.image;
        const bool can_transition_depth_slices =
            CanTransitionDepthSlices(env.validator.extensions, image_state.GetImageType(), image_state.create_flags);
        const auto hazard = access_context.DetectImageBarrierHazard(
            image_state, image_barrier.barrier.src_exec_scope.exec_scope, image_barrier.barrier.src_access_scope,
            image_barrier.subresource_range, can_transition_depth_slices, AccessContext::kDetectAll, env.queue_id);
        if (hazard.IsHazard() && !record_time_validated_tags.includes(hazard.Tag())) {
            if (replay_tag != kInvalidTag) {
                skip |= ReportReplayHazard(env, cb_context, hazard, replay_tag, image_state.Handle(), loc);
                continue;
            }
            const LogObjectList objlist = CommandObjectList(env, cb_context, image_state.Handle());
            const SyncValidator& sync_state = env.validator;
            const std::string resource_description = sync_state.FormatHandle(image_state.Handle());
            const std::string error =
                sync_state.error_messages_.ImageBarrierError(env, hazard, command, resource_description, image_barrier);
            skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
        }
    }
    return skip;
}

void PipelineBarrierCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_barrier) {
    for (const SyncBufferBarrier& buffer_barrier : barriers.buffer_barriers) {
        cb_context.AddCommandHandle(tag, buffer_barrier.buffer->Handle());
    }
    for (SyncImageBarrier& image_barrier : barriers.image_barriers) {
        if (image_barrier.layout_transition) {
            const ResourceUsageTagEx tag_ex = cb_context.AddCommandHandle(tag, image_barrier.image->Handle());
            image_barrier.handle_index = tag_ex.handle_index;
        }
    }
    if (apply_barrier) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), tag);
    }
}

void PipelineBarrierCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    // Unlike the legacy first-access replay, command replay builds the complete queue context
    // and therefore must materialize layout-transition writes at submit time.
    ApplyBarrier(env, access_context, barriers, tag, true);
}

bool EventCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), ResourceUsageRecord::kMaxIndex, {}, loc);
}

bool EventCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context, ResourceUsageTag tag,
                            const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    const Location command_loc = command == vvl::Func::Empty ? loc : Location(command);
    switch (type) {
        case Type::kSet:
            return ValidateCmdSetEvent(env, events.front(), exec_scope, tag, command_loc, record_time_validated_tags);
        case Type::kReset:
            return ValidateCmdResetEvent(env, events.front(), exec_scope, tag, command_loc, record_time_validated_tags);
        case Type::kWait: {
            bool skip = ValidateCmdWaitEvents(env, events, tag, command_loc, record_time_validated_tags);
            skip |= DetectCmdWaitEventsImageBarrierHazard(env, access_context, events, barrier_sets, tag, command_loc,
                                                          record_time_validated_tags);
            return skip;
        }
    }
    return false;
}

void EventCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_command) {
    if (apply_command) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), tag);
    }
}

void EventCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    switch (type) {
        case Type::kSet: {
            auto src_access_context = std::make_shared<AccessContext>(*access_context.validator);
            src_access_context->InitFrom(access_context);
            ApplyCmdSetEvent(env, events.front(), exec_scope, src_access_context, tag, command);
            break;
        }
        case Type::kReset:
            ApplyCmdResetEvent(env, events.front(), tag, command);
            break;
        case Type::kWait:
            ApplyCmdWaitEvents(env, access_context, events, barrier_sets, tag, command, true);
            break;
    }
}

bool NoOpCommand::Validate(const SyncEnvironment&, const AccessContext&, const CommandBufferContext&, vvl::Func,
                           const ResourceUsageRange&, const Location&, ResourceUsageTag) const {
    return false;
}

void NoOpCommand::Apply(SyncEnvironment&, AccessContext&, ResourceUsageTag) const {}

class CommandReplayContext {
  public:
    CommandReplayContext(SyncEnvironment& env, AccessContext& access_context) : env_(env), access_context_(access_context) {}

    AccessContext& CurrentAccessContext() {
        return render_pass_context_ ? render_pass_context_->CurrentContext() : access_context_;
    }

    RenderPassAccessContext* BeginRenderPass(const RenderPassCommand& command) {
        if (!command.render_pass) {
            return nullptr;
        }
        render_pass_context_ =
            std::make_unique<RenderPassAccessContext>(*command.render_pass, command.render_area, env_.queue_flags,
                                                      command.attachments, access_context_, command.render_pass_instance_id);
        return render_pass_context_.get();
    }

    RenderPassAccessContext* GetRenderPassContext() { return render_pass_context_.get(); }
    void EndRenderPass() { render_pass_context_.reset(); }
    AccessContext& GetExternalAccessContext() { return access_context_; }
    SyncEnvironment& GetSyncEnvironment() { return env_; }

  private:
    SyncEnvironment& env_;
    AccessContext& access_context_;
    std::unique_ptr<RenderPassAccessContext> render_pass_context_;
};

ResourceUsageTag RenderPassCommand::Record(CommandBufferContext& cb_context, bool apply_command) const {
    switch (type) {
        case Type::kBegin:
            if (render_pass) {
                return cb_context.RecordBeginRenderPass(command, *render_pass, render_area, attachments, apply_command);
            }
            break;
        case Type::kNext:
            return cb_context.RecordNextSubpass(command, apply_command);
        case Type::kEnd:
            return cb_context.RecordEndRenderPass(command, apply_command);
    }
    return ResourceUsageRecord::kMaxIndex;
}

bool RenderPassCommand::Validate(CommandReplayContext& replay_context, const CommandBufferContext& cb_context,
                                 const ResourceUsageRange& record_time_validated_tags, ResourceUsageTag replay_tag,
                                 const Location& loc) const {
    switch (type) {
        case Type::kBegin: {
            RenderPassAccessContext* rp_context = replay_context.BeginRenderPass(*this);
            return rp_context ? rp_context->ValidateBeginRenderPass(cb_context, replay_context.GetSyncEnvironment(),
                                                                    record_time_validated_tags, command, replay_tag, &loc)
                              : false;
        }
        case Type::kNext: {
            const RenderPassAccessContext* rp_context = replay_context.GetRenderPassContext();
            return rp_context ? rp_context->ValidateNextSubpass(cb_context, replay_context.GetSyncEnvironment(),
                                                                record_time_validated_tags, command, replay_tag, &loc)
                              : false;
        }
        case Type::kEnd: {
            const RenderPassAccessContext* rp_context = replay_context.GetRenderPassContext();
            return rp_context ? rp_context->ValidateEndRenderPass(cb_context, replay_context.GetSyncEnvironment(),
                                                                  record_time_validated_tags, command, replay_tag, &loc)
                              : false;
        }
    }
    return false;
}

void RenderPassCommand::Apply(CommandReplayContext& replay_context, ResourceUsageTag tag) const {
    RenderPassAccessContext* rp_context = replay_context.GetRenderPassContext();
    if (!rp_context && type == Type::kBegin) {
        rp_context = replay_context.BeginRenderPass(*this);
    }
    if (!rp_context) {
        return;
    }
    switch (type) {
        case Type::kBegin:
            rp_context->RecordBeginRenderPass(tag, tag + 1, replay_context.GetSyncEnvironment().queue_id);
            break;
        case Type::kNext:
            rp_context->RecordNextSubpass(tag, tag + 1, tag + 2, tag + 3, replay_context.GetSyncEnvironment().queue_id);
            break;
        case Type::kEnd:
            rp_context->RecordEndRenderPass(&replay_context.GetExternalAccessContext(), tag, tag + 1,
                                            replay_context.GetSyncEnvironment().queue_id);
            replay_context.EndRenderPass();
            break;
    }
}

bool ReplayRecordedCommands(SyncEnvironment& env, AccessContext& access_context, const CommandBufferContext& cb_context,
                            ResourceUsageTag base_tag, const ResourceUsageRange& record_time_validated_tags, const Location& loc) {
    bool skip = false;
    CommandReplayContext replay_context(env, access_context);
    const CommandData& command_data = cb_context.GetCommandData();
    for (const RecordedCommandEntry& entry : cb_context.GetRecordedCommands()) {
        const ResourceUsageTag replay_tag = base_tag + entry.tag;
        const vvl::Func command = cb_context.GetResourceUsageInfo(ResourceUsageTagEx{entry.tag}).command;
        auto replay_command = [&](const auto& resolved_command) {
            using CommandType = std::decay_t<decltype(resolved_command)>;
            if constexpr (std::is_same_v<CommandType, RenderPassCommand>) {
                skip |= resolved_command.Validate(replay_context, cb_context, record_time_validated_tags, entry.tag, loc);
                resolved_command.Apply(replay_context, replay_tag);
            } else if constexpr (std::is_same_v<CommandType, ResourceAccessCommand>) {
                AccessContext& current_access_context = replay_context.CurrentAccessContext();
                skip |= resolved_command.Validate(env, current_access_context, cb_context, command, record_time_validated_tags, loc,
                                                  entry.tag);
                resolved_command.Apply(env, current_access_context, replay_tag);
            } else if constexpr (std::is_same_v<CommandType, EventCommand>) {
                AccessContext& current_access_context = replay_context.CurrentAccessContext();
                skip |= resolved_command.Validate(env, current_access_context, replay_tag, record_time_validated_tags, loc);
                resolved_command.Apply(env, current_access_context, replay_tag);
            } else {
                AccessContext& current_access_context = replay_context.CurrentAccessContext();
                skip |= resolved_command.Validate(env, current_access_context, cb_context, command, record_time_validated_tags, loc,
                                                  entry.tag);
                resolved_command.Apply(env, current_access_context, replay_tag);
            }
        };
        std::visit(
            [&](const auto& recorded_command) {
                using CommandType = std::decay_t<decltype(recorded_command)>;
                if constexpr (std::is_same_v<CommandType, BufferCopyCommand::Storage>) {
                    const BufferCopyCommand resolved_command = recorded_command.MakeCommand(command_data);
                    AccessContext& current_access_context = replay_context.CurrentAccessContext();
                    skip |= resolved_command.Validate(env, current_access_context, cb_context, command,
                                                      record_time_validated_tags, loc, entry.tag);
                    const ResourceUsageTagEx src_tag_ex{replay_tag, recorded_command.src_handle_index};
                    const ResourceUsageTagEx dst_tag_ex{replay_tag, recorded_command.dst_handle_index};
                    resolved_command.Apply(env, current_access_context, src_tag_ex, dst_tag_ex);
                } else if constexpr (std::is_same_v<CommandType, BufferAccessCommand::Storage>) {
                    const BufferAccessCommand resolved_command = recorded_command.MakeCommand(command_data);
                    AccessContext& current_access_context = replay_context.CurrentAccessContext();
                    skip |= resolved_command.Validate(env, current_access_context, cb_context, command,
                                                      record_time_validated_tags, loc, entry.tag);
                    resolved_command.Apply(env, current_access_context,
                                           ResourceUsageTagEx{replay_tag, recorded_command.handle_index});
                } else if constexpr (std::is_same_v<CommandType, NoOpCommand>) {
                    replay_command(recorded_command);
                } else {
                    replay_command(recorded_command.MakeCommand(command_data));
                }
            },
            entry.command);
    }
    return skip;
}

void ApplyRecordedCommands(SyncEnvironment& env, AccessContext& access_context, const CommandBufferContext& cb_context,
                           ResourceUsageTag base_tag) {
    const CommandData& command_data = cb_context.GetCommandData();
    for (const RecordedCommandEntry& entry : cb_context.GetRecordedCommands()) {
        std::visit(
            [&](const auto& recorded_command) {
                using CommandType = std::decay_t<decltype(recorded_command)>;
                // This path flattens a secondary command buffer into its caller. A valid secondary cannot own render pass
                // structural commands; render-pass-continue secondary accesses are represented by the other command types.
                if constexpr (std::is_same_v<CommandType, BufferCopyCommand::Storage>) {
                    const BufferCopyCommand resolved_command = recorded_command.MakeCommand(command_data);
                    const ResourceUsageTag replay_tag = base_tag + entry.tag;
                    const ResourceUsageTagEx src_tag_ex{replay_tag, recorded_command.src_handle_index};
                    const ResourceUsageTagEx dst_tag_ex{replay_tag, recorded_command.dst_handle_index};
                    resolved_command.Apply(env, access_context, src_tag_ex, dst_tag_ex);
                } else if constexpr (std::is_same_v<CommandType, BufferAccessCommand::Storage>) {
                    const BufferAccessCommand resolved_command = recorded_command.MakeCommand(command_data);
                    resolved_command.Apply(env, access_context,
                                           ResourceUsageTagEx{base_tag + entry.tag, recorded_command.handle_index});
                } else if constexpr (std::is_same_v<CommandType, NoOpCommand>) {
                    recorded_command.Apply(env, access_context, base_tag + entry.tag);
                } else {
                    const auto& resolved_command = recorded_command.MakeCommand(command_data);
                    using ResolvedCommandType = std::decay_t<decltype(resolved_command)>;
                    if constexpr (!std::is_same_v<ResolvedCommandType, RenderPassCommand>) {
                        resolved_command.Apply(env, access_context, base_tag + entry.tag);
                    }
                }
            },
            entry.command);
    }
}

}  // namespace syncval
