/* Copyright (c) 2026 The Khronos Group Inc.
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
#include "sync/sync_access_context.h"
#include "sync/sync_command_buffer.h"
#include "sync/sync_event.h"
#include "sync/sync_image.h"
#include "sync/sync_render_pass.h"
#include "sync/sync_validation.h"
#include "state_tracker/buffer_state.h"
#include "state_tracker/descriptor_sets.h"
#include "state_tracker/image_state.h"
#include "state_tracker/render_pass_state.h"
#include "state_tracker/pipeline_state.h"
#include "state_tracker/ray_tracing_state.h"
#include "sync/sync_reporting.h"
#include "utils/image_utils.h"
#include "containers/small_vector.h"

namespace syncval {

struct CommandReplayContext {
    CommandReplayContext(SyncEnvironment& env, AccessContext& destination_access_context, ResourceUsageTag base_tag)
        : env(env), destination_access_context(destination_access_context), render_pass_instance_offset(uint32_t(base_tag)) {}

    AccessContext& CurrentAccessContext() {
        return render_pass_context ? render_pass_context->CurrentContext() : destination_access_context;
    }
    void BeginRenderPass(const BeginRenderPassCommand& command) {
        render_pass_context.emplace(command.render_pass, command.render_area, env.queue_flags, command.attachment_views,
                                    destination_access_context, command.render_pass_instance_id + render_pass_instance_offset,
                                    env.queue_id);
    }
    void NextSubpass() { render_pass_context->AdvanceSubpass(); }
    void EndRenderPass() { render_pass_context.reset(); }

    SyncEnvironment& env;
    AccessContext& destination_access_context;
    const uint32_t render_pass_instance_offset;
    std::optional<RenderPassAccessContext> render_pass_context;
};

bool ReplayCommands(SyncEnvironment& env, AccessContext& destination_access_context, const CommandBufferContext& cb_context,
                    ResourceUsageTag base_tag, const Location& loc) {
    bool skip = false;
    const CommandData& command_data = cb_context.GetCommandData();
    CommandReplayContext replay_context(env, destination_access_context, base_tag);

    for (const CommandEntry& entry : cb_context.GetCommands()) {
        const ResourceUsageTag tag = base_tag + entry.tag;
        std::visit(
            [&](const auto& storage) {
                bool command_skip = false;
                auto command = storage.MakeCommand(command_data);
                using CommandType = std::decay_t<decltype(command)>;
                if constexpr (std::is_same_v<CommandType, ResourceAccessCommand>) {
                    command.render_pass_instance_offset = replay_context.render_pass_instance_offset;
                } else if constexpr (std::is_same_v<CommandType, ShaderAccessCommand>) {
                    command.additional_accesses.render_pass_instance_offset = replay_context.render_pass_instance_offset;
                }

                AccessContext& access_context = replay_context.CurrentAccessContext();

                if constexpr (std::is_same_v<CommandType, BeginRenderPassCommand>) {
                    command_skip = command.Validate(env, access_context, cb_context, entry.tag, loc);
                    replay_context.BeginRenderPass(command);
                    command.Apply(env, tag, *replay_context.render_pass_context);
                } else if constexpr (std::is_same_v<CommandType, NextSubpassCommand>) {
                    command_skip = command.Validate(env, *replay_context.render_pass_context, cb_context, entry.tag, loc);
                    replay_context.NextSubpass();
                    command.Apply(env, tag, *replay_context.render_pass_context);
                } else if constexpr (std::is_same_v<CommandType, EndRenderPassCommand>) {
                    command_skip = command.Validate(env, *replay_context.render_pass_context, cb_context, entry.tag, loc);
                    command.Apply(env, tag, *replay_context.render_pass_context, destination_access_context);
                    replay_context.EndRenderPass();
                } else if constexpr (std::is_same_v<CommandType, EventCommand>) {
                    // Event scopes use tags in the destination context.
                    command_skip = command.Validate(env, access_context, cb_context, tag, loc);
                    command.Apply(env, tag, access_context);
                } else {
                    command_skip = command.Validate(env, access_context, cb_context, entry.tag, loc);
                    command.Apply(env, tag, access_context);
                }
                skip |= command_skip;
            },
            entry.storage);
    }
    return skip;
}

EventCommand EventCommand::Storage::MakeCommand(const CommandData& command_data) const {
    vvl::span<const std::shared_ptr<const vvl::Event>> event_span;
    if (event_count != 0) {
        event_span = vvl::make_span(&command_data.events[first_event], event_count);
    }
    vvl::span<const BarrierSet> barrier_span;
    if (barrier_set_count != 0) {
        barrier_span = vvl::make_span(&command_data.event_barrier_sets[first_barrier_set], barrier_set_count);
    }
    return {type, event_span, exec_scope, barrier_span, command};
}

EventCommand::Storage EventCommand::MakeStorage(CommandData& command_data) const {
    const uint32_t first_event = uint32_t(command_data.events.size());
    const uint32_t event_count = uint32_t(events.size());
    if (!events.empty()) {
        command_data.events.insert(command_data.events.end(), events.begin(), events.end());
    }

    const uint32_t first_barrier_set = uint32_t(command_data.event_barrier_sets.size());
    const uint32_t barrier_set_count = uint32_t(barrier_sets.size());
    if (!barrier_sets.empty()) {
        command_data.event_barrier_sets.insert(command_data.event_barrier_sets.end(), barrier_sets.begin(), barrier_sets.end());
    }
    return {type, first_event, event_count, exec_scope, first_barrier_set, barrier_set_count, command};
}

uint32_t CommandData::AddBuffer(const vvl::Buffer& buffer) {
    const uint32_t index = uint32_t(buffers.size());
    buffers.emplace_back(std::static_pointer_cast<const vvl::Buffer>(buffer.shared_from_this()));
    return index;
}

uint32_t CommandData::AddImage(const vvl::Image& image) {
    const uint32_t index = uint32_t(images.size());
    images.emplace_back(std::static_pointer_cast<const vvl::Image>(image.shared_from_this()));
    return index;
}

void CommandData::AddImageView(const vvl::ImageView& image_view) {
    image_views.emplace_back(std::static_pointer_cast<const vvl::ImageView>(image_view.shared_from_this()));
}

void CommandData::AddPipeline(const vvl::Pipeline& pipeline) {
    pipelines.emplace_back(std::static_pointer_cast<const vvl::Pipeline>(pipeline.shared_from_this()));
}

void CommandData::AddDescriptorSet(const vvl::DescriptorSet& descriptor_set) {
    descriptor_sets.emplace_back(std::static_pointer_cast<const vvl::DescriptorSet>(descriptor_set.shared_from_this()));
}

void CommandData::AddAccelerationStructure(const vvl::AccelerationStructureKHR& acceleration_structure) {
    acceleration_structures.emplace_back(
        std::static_pointer_cast<const vvl::AccelerationStructureKHR>(acceleration_structure.shared_from_this()));
}

ResourceAccessCommand ResourceAccessCommand::Storage::MakeCommand(const CommandData& command_data) const {
    vvl::span<const Access> access_span;
    if (access_count != 0) {
        access_span = vvl::make_span(&command_data.resource_accesses[first_access], access_count);
    }
    return {access_span};
}

ResourceAccessCommand::Storage ResourceAccessCommand::MakeStorage(CommandData& command_data) const {
    for (const Access& access : accesses) {
        std::visit(
            [&](const auto& value) {
                using AccessType = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    if (value.buffer) command_data.AddBuffer(*value.buffer);
                    if (value.pipeline) command_data.AddPipeline(*value.pipeline);
                    if (value.acceleration_structure_info && value.acceleration_structure_info->acceleration_structure) {
                        command_data.AddAccelerationStructure(*value.acceleration_structure_info->acceleration_structure);
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    if (value.image_view) command_data.AddImageView(*value.image_view);
                } else if constexpr (std::is_same_v<AccessType, ImageRangeAccess>) {
                    if (value.image) command_data.AddImage(*value.image);
                }
            },
            access);
    }
    const uint32_t first_access = uint32_t(command_data.resource_accesses.size());
    const uint32_t access_count = uint32_t(accesses.size());
    if (!accesses.empty()) {
        command_data.resource_accesses.insert(command_data.resource_accesses.end(), accesses.begin(), accesses.end());
    }
    return {first_access, access_count};
}

uint32_t CommandData::AddRenderPass(const vvl::RenderPass& render_pass) {
    const uint32_t index = uint32_t(render_passes.size());
    render_passes.emplace_back(std::static_pointer_cast<const vvl::RenderPass>(render_pass.shared_from_this()));
    return index;
}

BufferCopyCommand BufferCopyCommand::Storage::MakeCommand(const CommandData& command_data) const {
    const vvl::Buffer& src_buffer = *command_data.buffers[src_buffer_index];
    const vvl::Buffer& dst_buffer = *command_data.buffers[dst_buffer_index];
    vvl::span<const BufferCopyRegion> regions;
    if (region_count != 0) {
        regions = vvl::make_span(&command_data.buffer_copy_regions[first_region], region_count);
    }
    return {src_buffer, dst_buffer, regions, src_handle_index, dst_handle_index};
}

BufferCopyCommand::Storage BufferCopyCommand::MakeStorage(CommandData& command_data) const {
    const uint32_t src_buffer_index = command_data.AddBuffer(src_buffer);
    const uint32_t dst_buffer_index = command_data.AddBuffer(dst_buffer);

    const uint32_t first_region = uint32_t(command_data.buffer_copy_regions.size());
    const uint32_t region_count = uint32_t(regions.size());
    command_data.buffer_copy_regions.insert(command_data.buffer_copy_regions.end(), regions.begin(), regions.end());

    return {src_buffer_index, dst_buffer_index, first_region, region_count, src_handle_index, dst_handle_index};
}

BufferAccessCommand BufferAccessCommand::Storage::MakeCommand(const CommandData& command_data) const {
    return {*command_data.buffers[buffer_index], access_index, range, flags, query_pool, resource_name, handle_index};
}

BufferAccessCommand::Storage BufferAccessCommand::MakeStorage(CommandData& command_data) const {
    return {command_data.AddBuffer(buffer), access_index, range, flags, query_pool, resource_name, handle_index};
}

bool BufferAccessCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    const bool is_marker = (flags & SyncFlag::kMarker) != 0;
    const AccessContext& access_context = is_marker ? cb_context.GetCurrentAccessContext() : cb_context.GetCbAccessContext();
    return Validate(cb_context.GetSyncEnvironment(), access_context, cb_context, kInvalidTag, loc);
}

bool BufferAccessCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                   const CommandBufferContext& cb_context, ResourceUsageTag replay_tag, const Location& loc) const {
    const bool is_marker = (flags & SyncFlag::kMarker) != 0;
    const HazardResult hazard =
        is_marker ? access_context.DetectMarkerHazard(buffer, range) : access_context.DetectHazard(buffer, access_index, range);
    if (!hazard.IsHazard()) {
        return false;
    }

    const SyncValidator& validator = env.validator;
    LogObjectList objlist;
    if (replay_tag == kInvalidTag && is_marker) {
        objlist.add(buffer.Handle());
    } else {
        const VulkanTypedHandle query_pool_handle =
            query_pool != VK_NULL_HANDLE ? VulkanTypedHandle(query_pool, kVulkanObjectTypeQueryPool) : NullVulkanTypedHandle;
        objlist = BaseObjectList(env, cb_context, query_pool_handle);
        objlist.add(buffer.Handle());
    }
    const std::string resource_description = resource_name + validator.FormatHandle(buffer.Handle());
    const std::string error =
        validator.error_messages_.BufferError(env, hazard, cb_context, replay_tag, loc, resource_description, range);
    return validator.SyncError(hazard.Hazard(), objlist, loc, error);
}

void BufferAccessCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const {
    access_context.UpdateAccessState(buffer, access_index, range, ResourceUsageTagEx{tag, handle_index}, flags, env.queue_id);
}

bool BufferCopyCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), cb_context, kInvalidTag, loc);
}

bool BufferCopyCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                 const CommandBufferContext& cb_context, ResourceUsageTag replay_tag, const Location& loc) const {
    bool skip = false;
    const SyncValidator& validator = env.validator;

    for (const auto [region_index, region] : vvl::enumerate(regions)) {
        const AccessRange src_range = MakeRange(src_buffer, region.src_offset, region.size);
        auto src_hazard = access_context.DetectHazard(src_buffer, SYNC_COPY_TRANSFER_READ, src_range);
        if (src_hazard.IsHazard()) {
            const LogObjectList objlist = BaseObjectList(env, cb_context, src_buffer.Handle());
            const std::string resource_description = validator.FormatHandle(src_buffer);
            const std::string error = validator.error_messages_.BufferCopyError(
                env, src_hazard, cb_context, replay_tag, loc, resource_description, uint32_t(region_index), src_range);
            skip |= validator.SyncError(src_hazard.Hazard(), objlist, loc, error);
        }
        const AccessRange dst_range = MakeRange(dst_buffer, region.dst_offset, region.size);
        auto dst_hazard = access_context.DetectHazard(dst_buffer, SYNC_COPY_TRANSFER_WRITE, dst_range);
        if (dst_hazard.IsHazard()) {
            const LogObjectList objlist = BaseObjectList(env, cb_context, dst_buffer.Handle());
            const std::string resource_description = validator.FormatHandle(dst_buffer);
            const std::string error = validator.error_messages_.BufferCopyError(
                env, dst_hazard, cb_context, replay_tag, loc, resource_description, uint32_t(region_index), dst_range);
            skip |= validator.SyncError(dst_hazard.Hazard(), objlist, loc, error);
        }
        if (skip) {
            break;
        }
    }
    return skip;
}

void BufferCopyCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const {
    const ResourceUsageTagEx src_tag_ex{tag, src_handle_index};
    const ResourceUsageTagEx dst_tag_ex{tag, dst_handle_index};

    for (const BufferCopyRegion& region : regions) {
        const AccessRange src_range = MakeRange(src_buffer, region.src_offset, region.size);
        const AccessRange dst_range = MakeRange(dst_buffer, region.dst_offset, region.size);

        access_context.UpdateAccessState(src_buffer, SYNC_COPY_TRANSFER_READ, src_range, src_tag_ex, 0, env.queue_id);
        access_context.UpdateAccessState(dst_buffer, SYNC_COPY_TRANSFER_WRITE, dst_range, dst_tag_ex, 0, env.queue_id);
    }
}

ImageCopyCommand ImageCopyCommand::Storage::MakeCommand(const CommandData& command_data) const {
    const vvl::Image& src_image = *command_data.images[src_image_index];
    const vvl::Image& dst_image = *command_data.images[dst_image_index];
    vvl::span<const VkImageCopy> regions;
    if (region_count != 0) {
        regions = vvl::make_span(&command_data.image_copy_regions[first_region], region_count);
    }
    return {src_image, dst_image, regions, src_handle_index, dst_handle_index};
}

ImageCopyCommand::Storage ImageCopyCommand::MakeStorage(CommandData& command_data) const {
    const uint32_t src_image_index = command_data.AddImage(src_image);
    const uint32_t dst_image_index = command_data.AddImage(dst_image);

    const uint32_t first_region = uint32_t(command_data.image_copy_regions.size());
    const uint32_t region_count = uint32_t(regions.size());
    command_data.image_copy_regions.insert(command_data.image_copy_regions.end(), regions.begin(), regions.end());

    return {src_image_index, dst_image_index, first_region, region_count, src_handle_index, dst_handle_index};
}

bool ImageCopyCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), cb_context, kInvalidTag, loc);
}

bool ImageCopyCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                const CommandBufferContext& cb_context, ResourceUsageTag replay_tag, const Location& loc) const {
    bool skip = false;
    const SyncValidator& validator = env.validator;

    for (const auto [region_index, region] : vvl::enumerate(regions)) {
        auto src_hazard = access_context.DetectHazard(src_image, RangeFromLayers(region.srcSubresource), region.srcOffset,
                                                      region.extent, SYNC_COPY_TRANSFER_READ);
        if (src_hazard.IsHazard()) {
            const LogObjectList objlist = BaseObjectList(env, cb_context, src_image.Handle());
            const std::string resource_description = validator.FormatHandle(src_image);
            const std::string error = validator.error_messages_.ImageCopyResolveBlitError(
                env, src_hazard, cb_context, replay_tag, loc, resource_description, uint32_t(region_index), region.srcOffset,
                region.extent, region.srcSubresource);
            skip |= validator.SyncError(src_hazard.Hazard(), objlist, loc, error);
        }
        auto dst_hazard = access_context.DetectHazard(dst_image, RangeFromLayers(region.dstSubresource), region.dstOffset,
                                                      region.extent, SYNC_COPY_TRANSFER_WRITE);
        if (dst_hazard.IsHazard()) {
            const LogObjectList objlist = BaseObjectList(env, cb_context, dst_image.Handle());
            const std::string resource_description = validator.FormatHandle(dst_image);
            const std::string error = validator.error_messages_.ImageCopyResolveBlitError(
                env, dst_hazard, cb_context, replay_tag, loc, resource_description, uint32_t(region_index), region.dstOffset,
                region.extent, region.dstSubresource);
            skip |= validator.SyncError(dst_hazard.Hazard(), objlist, loc, error);
        }
        if (skip) {
            break;
        }
    }
    return skip;
}

static void UpdateImageAccessState(AccessContext& access_context, const vvl::Image& image, SyncAccessIndex current_usage,
                                   const VkImageSubresourceRange& subresource_range, const VkOffset3D& offset,
                                   const VkExtent3D& extent, ResourceUsageTagEx tag_ex, QueueId queue_id) {
    const auto& sub_state = SubState(image);
    ImageRangeGen range_gen = sub_state.MakeImageRangeGen(subresource_range, offset, extent, false);
    access_context.UpdateAccessState(range_gen, current_usage, tag_ex, 0, queue_id);
}

void ImageCopyCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const {
    const ResourceUsageTagEx src_tag_ex{tag, src_handle_index};
    const ResourceUsageTagEx dst_tag_ex{tag, dst_handle_index};

    for (const VkImageCopy& region : regions) {
        UpdateImageAccessState(access_context, src_image, SYNC_COPY_TRANSFER_READ, RangeFromLayers(region.srcSubresource),
                               region.srcOffset, region.extent, src_tag_ex, env.queue_id);
        UpdateImageAccessState(access_context, dst_image, SYNC_COPY_TRANSFER_WRITE, RangeFromLayers(region.dstSubresource),
                               region.dstOffset, region.extent, dst_tag_ex, env.queue_id);
    }
}

bool ResourceAccessCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, kInvalidTag, loc);
}

bool ResourceAccessCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                     const CommandBufferContext& cb_context, ResourceUsageTag replay_tag,
                                     const Location& loc) const {
    bool skip = false;
    const SyncValidator& validator = env.validator;

    for (const Access& access : accesses) {
        std::visit(
            [&](const auto& value) {
                using AccessType = std::decay_t<decltype(value)>;
                HazardResult hazard;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    if (!value.buffer) return;
                    hazard = (value.flags & SyncFlag::kMarker)
                                 ? access_context.DetectMarkerHazard(*value.buffer, value.range)
                                 : access_context.DetectHazard(*value.buffer, value.access_index, value.range);
                } else if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    if (!value.image_view) return;
                    if (value.use_render_area || value.view_mask != 0) {
                        ImageRangeGen range_gen;
                        if (value.view_mask != 0) {
                            range_gen = MakeImageRangeGen(*value.image_view, value.view_mask, value.aspect_mask);
                        } else {
                            range_gen = MakeImageRangeGen(*value.image_view, value.offset, value.extent, value.aspect_mask);
                        }
                        hazard = access_context.DetectAttachmentHazard(range_gen, value.access_index,
                                                                       GetAttachmentAccess(value.attachment_access), env.queue_id);
                    } else {
                        hazard = access_context.DetectHazard(*value.image_view, value.access_index);
                    }
                } else {
                    if (!value.image) return;
                    const auto& sub_state = SubState(*value.image);
                    ImageRangeGen range_gen = value.use_offset_extent
                                                  ? sub_state.MakeImageRangeGen(value.subresource_range, value.offset, value.extent,
                                                                                value.is_depth_sliced)
                                                  : sub_state.MakeImageRangeGen(value.subresource_range, value.is_depth_sliced);
                    if (value.attachment_access.type == AttachmentAccessType::Empty) {
                        hazard = access_context.DetectHazard(range_gen, value.access_index);
                    } else {
                        hazard = access_context.DetectAttachmentHazard(range_gen, value.access_index,
                                                                       GetAttachmentAccess(value.attachment_access), env.queue_id);
                    }
                }
                if (!hazard.IsHazard()) return;

                VulkanTypedHandle resource_handle;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    resource_handle = value.buffer->Handle();
                } else if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    resource_handle = value.image_view->Handle();
                } else {
                    resource_handle = value.tag_handle;
                }
                const VulkanTypedHandle object_handle = resource_handle;

                LogObjectList objlist;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    if (replay_tag == kInvalidTag && value.legacy_record_time_object_only) {
                        objlist = LogObjectList(object_handle);
                    } else {
                        objlist = BaseObjectList(env, cb_context, object_handle);
                    }
                    if (value.pipeline) objlist.add(value.pipeline->Handle());
                } else if constexpr (std::is_same_v<AccessType, ImageRangeAccess>) {
                    if (replay_tag == kInvalidTag && value.error_type == ImageRangeAccess::ErrorType::kVideo) {
                        objlist = LogObjectList(object_handle);
                    } else {
                        objlist = BaseObjectList(env, cb_context, object_handle);
                    }
                } else {
                    objlist = BaseObjectList(env, cb_context, object_handle);
                    objlist.add(value.additional_object);
                }

                std::string error;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    const std::string resource_description = value.resource_name + validator.FormatHandle(resource_handle);
                    if (value.acceleration_structure_info && value.acceleration_structure_info->acceleration_structure) {
                        const auto& acceleration_structure = *value.acceleration_structure_info->acceleration_structure;
                        objlist.add(acceleration_structure.Handle());
                        error = validator.error_messages_.AccelerationStructureError(
                            env, hazard, cb_context, replay_tag, loc, resource_description, value.range,
                            acceleration_structure.VkHandle(), value.acceleration_structure_info->location);
                    } else {
                        error = validator.error_messages_.BufferError(env, hazard, cb_context, replay_tag, loc,
                                                                      resource_description, value.range);
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    std::string resource_description =
                        value.resource_description.empty() ? validator.FormatHandle(resource_handle) : value.resource_description;
                    if (!value.resource_description.empty() && value.additional_object != NullVulkanTypedHandle) {
                        resource_description += " (" + validator.FormatHandle(resource_handle) + ", " +
                                                validator.FormatHandle(value.additional_object) + ")";
                    }
                    error = validator.error_messages_.Error(env, hazard, cb_context, replay_tag, loc, resource_description,
                                                            value.message_type);
                } else {
                    switch (value.error_type) {
                        case ImageRangeAccess::ErrorType::kClearAttachment:
                            error = validator.error_messages_.ClearAttachmentError(env, hazard, cb_context, replay_tag, loc,
                                                                                   value.resource_description, value.clear_aspects,
                                                                                   value.clear_rect_index, value.clear_rect);
                            break;
                        case ImageRangeAccess::ErrorType::kVideo:
                            error = validator.error_messages_.VideoError(env, hazard, cb_context, replay_tag, loc,
                                                                         value.resource_description);
                            break;
                        case ImageRangeAccess::ErrorType::kGeneric:
                            error = validator.error_messages_.Error(env, hazard, cb_context, replay_tag, loc,
                                                                    value.resource_description, "ImageRangeAccessError");
                            break;
                    }
                }

                auto report_error = [&](const Location& error_loc) {
                    skip |= validator.SyncError(hazard.Hazard(), objlist, error_loc, error);
                };
                if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
                    if (replay_tag == kInvalidTag) {
                        switch (value.error_location) {
                            case ImageViewAccess::ErrorLocation::kColorAttachment:
                                report_error(loc.dot(vvl::Struct::VkRenderingAttachmentInfo, vvl::Field::pColorAttachments,
                                                     value.attachment_index)
                                                 .dot(vvl::Field::imageView));
                                return;
                            case ImageViewAccess::ErrorLocation::kDepthAttachment:
                                report_error(loc.dot(vvl::Struct::VkRenderingAttachmentInfo, vvl::Field::pDepthAttachment)
                                                 .dot(vvl::Field::imageView));
                                return;
                            case ImageViewAccess::ErrorLocation::kStencilAttachment:
                                report_error(loc.dot(vvl::Struct::VkRenderingAttachmentInfo, vvl::Field::pStencilAttachment)
                                                 .dot(vvl::Field::imageView));
                                return;
                            case ImageViewAccess::ErrorLocation::kNone:
                                break;
                        }
                    }
                }
                report_error(loc);
            },
            access);
    }
    return skip;
}

void ResourceAccessCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const {
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
                    if (!value.image_view) return;
                    if (value.use_render_area || value.view_mask != 0) {
                        ImageRangeGen range_gen;
                        if (value.view_mask != 0) {
                            range_gen = MakeImageRangeGen(*value.image_view, value.view_mask, value.aspect_mask);
                        } else {
                            range_gen = MakeImageRangeGen(*value.image_view, value.offset, value.extent, value.aspect_mask);
                        }
                        access_context.UpdateAttachmentAccessState(
                            range_gen, value.access_index, GetAttachmentAccess(value.attachment_access), tag_ex, env.queue_id);
                    } else {
                        ImageRangeGen range_gen = MakeImageRangeGen(*value.image_view);
                        access_context.UpdateAccessState(range_gen, value.access_index, tag_ex, 0, env.queue_id);
                    }
                } else {
                    if (!value.image) return;
                    const auto& sub_state = SubState(*value.image);
                    ImageRangeGen range_gen = value.use_offset_extent
                                                  ? sub_state.MakeImageRangeGen(value.subresource_range, value.offset, value.extent,
                                                                                value.is_depth_sliced)
                                                  : sub_state.MakeImageRangeGen(value.subresource_range, value.is_depth_sliced);
                    if (value.attachment_access.type == AttachmentAccessType::Empty) {
                        access_context.UpdateAccessState(range_gen, value.access_index, tag_ex, 0, env.queue_id);
                    } else {
                        access_context.UpdateAttachmentAccessState(
                            range_gen, value.access_index, GetAttachmentAccess(value.attachment_access), tag_ex, env.queue_id);
                    }
                }
            },
            access);
    }
}

ShaderAccessCommand ShaderAccessCommand::Storage::MakeCommand(const CommandData& command_data) const {
    vvl::span<const BufferAccess> buffer_accesses;
    if (buffer_access_count != 0) {
        buffer_accesses = vvl::make_span(&command_data.descriptor_buffer_accesses[first_buffer_access], buffer_access_count);
    }
    vvl::span<const ImageViewAccess> image_accesses;
    if (image_access_count != 0) {
        image_accesses = vvl::make_span(&command_data.descriptor_image_accesses[first_image_access], image_access_count);
    }
    return {pipeline, buffer_accesses, image_accesses, additional_accesses.MakeCommand(command_data)};
}

ShaderAccessCommand::Storage ShaderAccessCommand::MakeStorage(CommandData& command_data) const {
    if (pipeline) command_data.AddPipeline(*pipeline);
    small_vector<const vvl::DescriptorSet*, 4> descriptor_sets;
    auto retain_descriptor_set = [&](const vvl::DescriptorSet* descriptor_set) {
        if (std::find(descriptor_sets.begin(), descriptor_sets.end(), descriptor_set) == descriptor_sets.end()) {
            command_data.AddDescriptorSet(*descriptor_set);
            descriptor_sets.emplace_back(descriptor_set);
        }
    };
    for (const BufferAccess& access : buffer_accesses) {
        command_data.AddBuffer(*access.buffer);
        retain_descriptor_set(access.info.descriptor_set);
    }
    for (const ImageViewAccess& access : image_accesses) {
        command_data.AddImageView(*access.image_view);
        retain_descriptor_set(access.info.descriptor_set);
    }

    const uint32_t first_buffer_access = uint32_t(command_data.descriptor_buffer_accesses.size());
    const uint32_t buffer_access_count = uint32_t(buffer_accesses.size());
    if (!buffer_accesses.empty()) {
        command_data.descriptor_buffer_accesses.insert(command_data.descriptor_buffer_accesses.end(), buffer_accesses.begin(),
                                                       buffer_accesses.end());
    }
    const uint32_t first_image_access = uint32_t(command_data.descriptor_image_accesses.size());
    const uint32_t image_access_count = uint32_t(image_accesses.size());
    if (!image_accesses.empty()) {
        command_data.descriptor_image_accesses.insert(command_data.descriptor_image_accesses.end(), image_accesses.begin(),
                                                      image_accesses.end());
    }
    return {pipeline,           first_buffer_access, buffer_access_count,
            first_image_access, image_access_count,  additional_accesses.MakeStorage(command_data)};
}

bool ShaderAccessCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, kInvalidTag, loc);
}

bool ShaderAccessCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                   const CommandBufferContext& cb_context, ResourceUsageTag replay_tag, const Location& loc) const {
    bool skip = false;
    const SyncValidator& validator = env.validator;
    auto validate_access = [&](const auto& value) {
        using AccessType = std::decay_t<decltype(value)>;
        HazardResult hazard;
        if constexpr (std::is_same_v<AccessType, BufferAccess>) {
            hazard = access_context.DetectHazard(*value.buffer, value.access_index, value.range);
        } else {
            if (value.access_index == SYNC_FRAGMENT_SHADER_INPUT_ATTACHMENT_READ) {
                ImageRangeGen range_gen = MakeImageRangeGen(*value.image_view, value.offset, value.extent);
                hazard = access_context.DetectAttachmentHazard(
                    range_gen, value.access_index, additional_accesses.GetAttachmentAccess(value.attachment_access), env.queue_id);
            } else {
                hazard = access_context.DetectHazard(*value.image_view, value.access_index);
            }
        }
        if (!hazard.IsHazard()) return;

        const DescriptorInfo& info = value.info;
        VulkanTypedHandle object_handle = info.resource_handle;
        if constexpr (std::is_same_v<AccessType, BufferAccess>) {
            if (info.resource_handle.type == kVulkanObjectTypeAccelerationStructureKHR) {
                object_handle = value.buffer->Handle();
            }
        }
        LogObjectList objlist = BaseObjectList(env, cb_context, object_handle);
        objlist.add(pipeline->Handle());
        std::string resource_description = validator.FormatHandle(info.resource_handle);
        if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
            if (replay_tag != kInvalidTag && value.image_view->image_state) {
                resource_description += " (" + validator.FormatHandle(value.image_view->image_state->Handle()) + ")";
            }
        }

        std::string error;
        if constexpr (std::is_same_v<AccessType, ImageViewAccess>) {
            error = validator.error_messages_.ImageDescriptorError(
                env, hazard, cb_context, replay_tag, loc, resource_description, *pipeline, info.set, *info.descriptor_set,
                info.descriptor_type, info.binding, info.array_element, info.stage, value.image_layout);
        } else {
            if (info.resource_handle.type == kVulkanObjectTypeAccelerationStructureKHR) {
                error = validator.error_messages_.AccelerationStructureDescriptorError(
                    env, hazard, cb_context, replay_tag, loc, resource_description, *pipeline, info.set, *info.descriptor_set,
                    info.descriptor_type, info.binding, info.array_element, info.stage);
            } else {
                error = validator.error_messages_.BufferDescriptorError(
                    env, hazard, cb_context, replay_tag, loc, resource_description, *pipeline, info.set, *info.descriptor_set,
                    info.descriptor_type, info.binding, info.array_element, info.stage);
            }
        }
        skip |= validator.SyncError(hazard.Hazard(), objlist, loc, error);
    };
    for (const BufferAccess& access : buffer_accesses) {
        validate_access(access);
    }
    for (const ImageViewAccess& access : image_accesses) {
        validate_access(access);
    }
    skip |= additional_accesses.Validate(env, access_context, cb_context, replay_tag, loc);
    return skip;
}

void ShaderAccessCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const {
    for (const BufferAccess& access : buffer_accesses) {
        const ResourceUsageTagEx tag_ex{tag, access.handle_index};
        access_context.UpdateAccessState(*access.buffer, access.access_index, access.range, tag_ex, 0, env.queue_id);
    }
    for (const ImageViewAccess& access : image_accesses) {
        const ResourceUsageTagEx tag_ex{tag, access.handle_index};
        if (access.access_index == SYNC_FRAGMENT_SHADER_INPUT_ATTACHMENT_READ) {
            ImageRangeGen range_gen = MakeImageRangeGen(*access.image_view, access.offset, access.extent);
            access_context.UpdateAttachmentAccessState(range_gen, access.access_index,
                                                       additional_accesses.GetAttachmentAccess(access.attachment_access), tag_ex,
                                                       env.queue_id);
        } else {
            ImageRangeGen range_gen = MakeImageRangeGen(*access.image_view);
            access_context.UpdateAccessState(range_gen, access.access_index, tag_ex, 0, env.queue_id);
        }
    }
    additional_accesses.Apply(env, tag, access_context);
}

ImageTransferCommand ImageTransferCommand::Storage::MakeCommand(const CommandData& command_data) const {
    vvl::span<const Access> accesses;
    if (access_count != 0) {
        accesses = vvl::make_span(&command_data.image_transfer_accesses[first_access], access_count);
    }
    return {accesses};
}

ImageTransferCommand::Storage ImageTransferCommand::MakeStorage(CommandData& command_data) const {
    for (const Access& access : accesses) {
        std::visit(
            [&](const auto& value) {
                using AccessType = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    command_data.AddBuffer(*value.buffer);
                } else {
                    command_data.AddImage(*value.image);
                }
            },
            access);
    }
    const uint32_t first_access = uint32_t(command_data.image_transfer_accesses.size());
    const uint32_t access_count = uint32_t(accesses.size());
    if (!accesses.empty()) {
        command_data.image_transfer_accesses.insert(command_data.image_transfer_accesses.end(), accesses.begin(), accesses.end());
    }
    return {first_access, access_count};
}

namespace {

template <typename RegionType>
std::vector<ImageTransferCommand::Access> CollectBufferToImageCopyAccessesImpl(const vvl::Buffer* src_buffer,
                                                                               const vvl::Image* dst_image, uint32_t region_count,
                                                                               const RegionType* regions) {
    std::vector<ImageTransferCommand::Access> accesses;
    accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_buffer && dst_image) {
            const AccessRange range = MakeRange(region.bufferOffset, dst_image->GetBufferSizeFromCopyImage(region));
            accesses.emplace_back(ImageTransferCommand::BufferAccess{src_buffer, SYNC_COPY_TRANSFER_READ, range, region_index});
        }
        if (dst_image) {
            accesses.emplace_back(ImageTransferCommand::ImageAccess{dst_image, SYNC_COPY_TRANSFER_WRITE, region.imageSubresource,
                                                                    region.imageOffset, region.imageExtent, region_index});
        }
    }
    return accesses;
}

template <typename RegionType>
std::vector<ImageTransferCommand::Access> CollectImageToBufferCopyAccessesImpl(const vvl::Image* src_image,
                                                                               const vvl::Buffer* dst_buffer, uint32_t region_count,
                                                                               const RegionType* regions) {
    std::vector<ImageTransferCommand::Access> accesses;
    accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_image) {
            accesses.emplace_back(ImageTransferCommand::ImageAccess{src_image, SYNC_COPY_TRANSFER_READ, region.imageSubresource,
                                                                    region.imageOffset, region.imageExtent, region_index});
        }
        if (src_image && dst_buffer) {
            const AccessRange range = MakeRange(region.bufferOffset, src_image->GetBufferSizeFromCopyImage(region));
            accesses.emplace_back(ImageTransferCommand::BufferAccess{dst_buffer, SYNC_COPY_TRANSFER_WRITE, range, region_index});
        }
    }
    return accesses;
}

template <typename RegionType>
std::vector<ImageTransferCommand::Access> CollectImageBlitAccessesImpl(const vvl::Image* src_image, const vvl::Image* dst_image,
                                                                       uint32_t region_count, const RegionType* regions) {
    std::vector<ImageTransferCommand::Access> accesses;
    accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_image) {
            const VkOffset3D offset = {std::min(region.srcOffsets[0].x, region.srcOffsets[1].x),
                                       std::min(region.srcOffsets[0].y, region.srcOffsets[1].y),
                                       std::min(region.srcOffsets[0].z, region.srcOffsets[1].z)};
            const VkExtent3D extent = {uint32_t(std::abs(region.srcOffsets[1].x - region.srcOffsets[0].x)),
                                       uint32_t(std::abs(region.srcOffsets[1].y - region.srcOffsets[0].y)),
                                       uint32_t(std::abs(region.srcOffsets[1].z - region.srcOffsets[0].z))};
            accesses.emplace_back(ImageTransferCommand::ImageAccess{src_image, SYNC_BLIT_TRANSFER_READ, region.srcSubresource,
                                                                    offset, extent, region_index});
        }
        if (dst_image) {
            const VkOffset3D offset = {std::min(region.dstOffsets[0].x, region.dstOffsets[1].x),
                                       std::min(region.dstOffsets[0].y, region.dstOffsets[1].y),
                                       std::min(region.dstOffsets[0].z, region.dstOffsets[1].z)};
            const VkExtent3D extent = {uint32_t(std::abs(region.dstOffsets[1].x - region.dstOffsets[0].x)),
                                       uint32_t(std::abs(region.dstOffsets[1].y - region.dstOffsets[0].y)),
                                       uint32_t(std::abs(region.dstOffsets[1].z - region.dstOffsets[0].z))};
            accesses.emplace_back(ImageTransferCommand::ImageAccess{dst_image, SYNC_BLIT_TRANSFER_WRITE, region.dstSubresource,
                                                                    offset, extent, region_index});
        }
    }
    return accesses;
}

template <typename RegionType>
std::vector<ImageTransferCommand::Access> CollectImageResolveAccessesImpl(const vvl::Image* src_image, const vvl::Image* dst_image,
                                                                          uint32_t region_count, const RegionType* regions) {
    std::vector<ImageTransferCommand::Access> accesses;
    accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_image) {
            accesses.emplace_back(ImageTransferCommand::ImageAccess{src_image, SYNC_RESOLVE_TRANSFER_READ, region.srcSubresource,
                                                                    region.srcOffset, region.extent, region_index});
        }
        if (dst_image) {
            accesses.emplace_back(ImageTransferCommand::ImageAccess{dst_image, SYNC_RESOLVE_TRANSFER_WRITE, region.dstSubresource,
                                                                    region.dstOffset, region.extent, region_index});
        }
    }
    return accesses;
}

}  // namespace

std::vector<ImageTransferCommand::Access> CollectBufferToImageCopyAccesses(const vvl::Buffer* src_buffer,
                                                                           const vvl::Image* dst_image, uint32_t region_count,
                                                                           const VkBufferImageCopy* regions) {
    return CollectBufferToImageCopyAccessesImpl(src_buffer, dst_image, region_count, regions);
}

std::vector<ImageTransferCommand::Access> CollectBufferToImageCopyAccesses(const vvl::Buffer* src_buffer,
                                                                           const vvl::Image* dst_image, uint32_t region_count,
                                                                           const VkBufferImageCopy2* regions) {
    return CollectBufferToImageCopyAccessesImpl(src_buffer, dst_image, region_count, regions);
}

std::vector<ImageTransferCommand::Access> CollectImageToBufferCopyAccesses(const vvl::Image* src_image,
                                                                           const vvl::Buffer* dst_buffer, uint32_t region_count,
                                                                           const VkBufferImageCopy* regions) {
    return CollectImageToBufferCopyAccessesImpl(src_image, dst_buffer, region_count, regions);
}

std::vector<ImageTransferCommand::Access> CollectImageToBufferCopyAccesses(const vvl::Image* src_image,
                                                                           const vvl::Buffer* dst_buffer, uint32_t region_count,
                                                                           const VkBufferImageCopy2* regions) {
    return CollectImageToBufferCopyAccessesImpl(src_image, dst_buffer, region_count, regions);
}

std::vector<ImageTransferCommand::Access> CollectImageBlitAccesses(const vvl::Image* src_image, const vvl::Image* dst_image,
                                                                   uint32_t region_count, const VkImageBlit* regions) {
    return CollectImageBlitAccessesImpl(src_image, dst_image, region_count, regions);
}

std::vector<ImageTransferCommand::Access> CollectImageBlitAccesses(const vvl::Image* src_image, const vvl::Image* dst_image,
                                                                   uint32_t region_count, const VkImageBlit2* regions) {
    return CollectImageBlitAccessesImpl(src_image, dst_image, region_count, regions);
}

std::vector<ImageTransferCommand::Access> CollectImageResolveAccesses(const vvl::Image* src_image, const vvl::Image* dst_image,
                                                                      uint32_t region_count, const VkImageResolve* regions) {
    return CollectImageResolveAccessesImpl(src_image, dst_image, region_count, regions);
}

std::vector<ImageTransferCommand::Access> CollectImageResolveAccesses(const vvl::Image* src_image, const vvl::Image* dst_image,
                                                                      uint32_t region_count, const VkImageResolve2* regions) {
    return CollectImageResolveAccessesImpl(src_image, dst_image, region_count, regions);
}

std::vector<ImageTransferCommand::Access> CollectImageClearAccesses(const vvl::Image* image, uint32_t range_count,
                                                                    const VkImageSubresourceRange* ranges) {
    std::vector<ImageTransferCommand::Access> accesses;
    accesses.reserve(range_count);
    if (image) {
        for (const auto [range_index, range] : vvl::enumerate(ranges, range_count)) {
            accesses.emplace_back(ImageTransferCommand::ImageRangeAccess{image, SYNC_CLEAR_TRANSFER_WRITE, range, range_index});
        }
    }
    return accesses;
}

bool ImageTransferCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), cb_context, kInvalidTag, loc);
}

bool ImageTransferCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                    const CommandBufferContext& cb_context, ResourceUsageTag replay_tag,
                                    const Location& loc) const {
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
                    const auto hazard = access_context.DetectHazard(*value.buffer, value.access_index, value.range);
                    if (hazard.IsHazard()) {
                        hazard_region = value.region_index;
                        const auto objects = BaseObjectList(env, cb_context, value.buffer->Handle());
                        const auto error = env.validator.error_messages_.BufferCopyError(
                            env, hazard, cb_context, replay_tag, loc, env.validator.FormatHandle(value.buffer->Handle()),
                            value.region_index, value.range);
                        skip |= env.validator.SyncError(hazard.Hazard(), objects, loc, error);
                    }
                } else if constexpr (std::is_same_v<AccessType, ImageAccess>) {
                    const auto hazard = access_context.DetectHazard(*value.image, RangeFromLayers(value.subresource), value.offset,
                                                                    value.extent, value.access_index);
                    if (hazard.IsHazard()) {
                        hazard_region = value.region_index;
                        const auto objects = BaseObjectList(env, cb_context, value.image->Handle());
                        const auto error = env.validator.error_messages_.ImageCopyResolveBlitError(
                            env, hazard, cb_context, replay_tag, loc, env.validator.FormatHandle(value.image->Handle()),
                            value.region_index, value.offset, value.extent, value.subresource);
                        skip |= env.validator.SyncError(hazard.Hazard(), objects, loc, error);
                    }
                } else {
                    const auto hazard = access_context.DetectHazard(*value.image, value.subresource_range, value.access_index);
                    if (hazard.IsHazard()) {
                        hazard_region = value.range_index;
                        const auto objects = BaseObjectList(env, cb_context, value.image->Handle());
                        const auto error = env.validator.error_messages_.ImageClearError(
                            env, hazard, cb_context, replay_tag, loc, env.validator.FormatHandle(value.image->Handle()),
                            value.range_index, value.subresource_range);
                        skip |= env.validator.SyncError(hazard.Hazard(), objects, loc, error);
                    }
                }
            },
            access);
    }
    return skip;
}

void ImageTransferCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const {
    for (const Access& access : accesses) {
        std::visit(
            [&](const auto& value) {
                using AccessType = std::decay_t<decltype(value)>;
                const ResourceUsageTagEx tag_ex{tag, value.handle_index};
                if constexpr (std::is_same_v<AccessType, BufferAccess>) {
                    access_context.UpdateAccessState(*value.buffer, value.access_index, value.range, tag_ex, 0, env.queue_id);
                } else if constexpr (std::is_same_v<AccessType, ImageAccess>) {
                    ImageRangeGen range_gen =
                        SubState(*value.image)
                            .MakeImageRangeGen(RangeFromLayers(value.subresource), value.offset, value.extent, false);
                    access_context.UpdateAccessState(range_gen, value.access_index, tag_ex, 0, env.queue_id);
                } else {
                    ImageRangeGen range_gen = SubState(*value.image).MakeImageRangeGen(value.subresource_range, false);
                    access_context.UpdateAccessState(range_gen, value.access_index, tag_ex, 0, env.queue_id);
                }
            },
            access);
    }
}

BarrierCommand BarrierCommand::Storage::MakeCommand(const CommandData& command_data) const {
    return BarrierCommand{command_data.barrier_sets[barrier_set_index]};
}

BarrierCommand::Storage BarrierCommand::MakeStorage(CommandData& command_data) const {
    const uint32_t barrier_set_index = uint32_t(command_data.barrier_sets.size());
    command_data.barrier_sets.emplace_back(barrier_set);
    return {barrier_set_index};
}

bool BarrierCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, kInvalidTag, loc);
}

bool BarrierCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                              const CommandBufferContext& cb_context, ResourceUsageTag replay_tag, const Location& loc) const {
    bool skip = false;
    const SyncValidator& validator = env.validator;

    for (const auto& image_barrier : barrier_set.image_barriers) {
        if (!image_barrier.layout_transition) {
            // The only accesses that originate from the pipeline barrier are layout transitions
            continue;
        }
        const vvl::Image& image_state = *image_barrier.image;
        const bool can_transition_depth_slices =
            CanTransitionDepthSlices(validator.extensions, image_state.GetImageType(), image_state.create_flags);

        const auto hazard = access_context.DetectImageBarrierHazard(
            image_state, image_barrier.barrier.src_exec_scope.exec_scope, image_barrier.barrier.src_access_scope,
            image_barrier.subresource_range, can_transition_depth_slices, AccessContext::kDetectAll, env.queue_id);

        if (hazard.IsHazard()) {
            const LogObjectList objlist = BaseObjectList(env, cb_context, image_state.Handle());
            const std::string resource_description = validator.FormatHandle(image_state.Handle());
            const std::string error = validator.error_messages_.ImageBarrierError(env, hazard, cb_context, replay_tag, loc,
                                                                                  resource_description, image_barrier);
            skip |= validator.SyncError(hazard.Hazard(), objlist, loc, error);
        }
    }
    return skip;
}

void BarrierCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const {
    ApplyBarrier(env, access_context, barrier_set, tag, true);
}

BeginRenderPassCommand BeginRenderPassCommand::Storage::MakeCommand(const CommandData& command_data) const {
    const vvl::RenderPass& render_pass = *command_data.render_passes[render_pass_index];
    vvl::span<const std::shared_ptr<const vvl::ImageView>> attachment_views;
    if (attachment_count != 0) {
        attachment_views = vvl::make_span(&command_data.image_views[first_attachment_view_index], attachment_count);
    }
    return BeginRenderPassCommand{render_pass, attachment_views, render_area, render_pass_instance_id};
}

BeginRenderPassCommand::Storage BeginRenderPassCommand::MakeStorage(CommandData& command_data) const {
    const uint32_t render_pass_index = command_data.AddRenderPass(render_pass);
    const uint32_t first_attachment = uint32_t(command_data.image_views.size());
    const uint32_t attachment_count = uint32_t(attachment_views.size());
    command_data.image_views.insert(command_data.image_views.end(), attachment_views.begin(), attachment_views.end());
    return {render_pass_index, first_attachment, attachment_count, render_area, render_pass_instance_id};
}

bool BeginRenderPassCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), cb_context, kInvalidTag, loc);
}

bool BeginRenderPassCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                      const CommandBufferContext& cb_context, ResourceUsageTag replay_tag,
                                      const Location& loc) const {
    bool skip = false;
    const uint32_t view_mask = render_pass.create_info.pSubpasses[0].viewMask;

    // Build temp subpass-0 context for simulating initial layout transitions.
    // NOTE: nullptr contexts parameter is safe for subpass zero:
    //  a) its non-external dependencies map is empty (an entry is created
    //     for src_subpass < dst_subpass but dst_subpass is zero)
    //  b) async list for subpass 0 is also empty (needs prev subpass too)
    AccessContext temp_context(env.validator);
    temp_context.InitFrom(0, env.queue_flags, render_pass.subpass_dependency_infos, nullptr, access_context, env.queue_id);

    // Validation runs before the render-pass context exists, so create the attachment view generators locally
    const AttachmentViewGenVector view_gens = RenderPassAccessContext::CreateAttachmentViewGen(render_area, attachment_views);

    skip |= RenderPassAccessContext::ValidateLayoutTransitions(env, temp_context, render_pass, render_pass_instance_id, 0,
                                                               view_mask, view_gens, cb_context, replay_tag, loc);
    if (!skip) {
        // Simulate initial layout transitions in the temporary context before validating load operations
        RenderPassAccessContext::RecordLayoutTransitions(render_pass, 0, view_gens, kInvalidTag, temp_context);

        skip |= RenderPassAccessContext::ValidateLoadOperation(env, temp_context, render_pass, render_pass_instance_id, 0,
                                                               view_mask, view_gens, cb_context, replay_tag, loc);
    }
    return skip;
}

void BeginRenderPassCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, RenderPassAccessContext& rp_context) const {
    const ResourceUsageTag transition_tag = tag;
    const ResourceUsageTag load_op_tag = tag + 1;
    rp_context.RecordBeginRenderPass(transition_tag, load_op_tag, env.queue_id);
}

bool NextSubpassCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    const RenderPassAccessContext* render_pass_context = cb_context.GetCurrentRenderPassContext();
    if (!render_pass_context) {
        return false;
    }
    return Validate(cb_context.GetSyncEnvironment(), *render_pass_context, cb_context, kInvalidTag, loc);
}
bool NextSubpassCommand::Validate(const SyncEnvironment& env, const RenderPassAccessContext& render_pass_context,
                                  const CommandBufferContext& cb_context, ResourceUsageTag replay_tag, const Location& loc) const {
    return render_pass_context.ValidateNextSubpass(env, cb_context, replay_tag, loc);
}

void NextSubpassCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, RenderPassAccessContext& rp_context) const {
    const ResourceUsageTag resolve_tag = tag;
    const ResourceUsageTag store_tag = tag + 1;
    const ResourceUsageTag transition_tag = tag + 2;
    const ResourceUsageTag load_tag = tag + 3;
    rp_context.RecordNextSubpass(resolve_tag, store_tag, transition_tag, load_tag, env.queue_id);
}

bool EndRenderPassCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    const RenderPassAccessContext* render_pass_context = cb_context.GetCurrentRenderPassContext();
    if (!render_pass_context) {
        return false;
    }
    return Validate(cb_context.GetSyncEnvironment(), *render_pass_context, cb_context, kInvalidTag, loc);
}

bool EndRenderPassCommand::Validate(const SyncEnvironment& env, const RenderPassAccessContext& render_pass_context,
                                    const CommandBufferContext& cb_context, ResourceUsageTag replay_tag,
                                    const Location& loc) const {
    return render_pass_context.ValidateEndRenderPass(env, cb_context, replay_tag, loc);
}

void EndRenderPassCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, RenderPassAccessContext& rp_context,
                                 AccessContext& external_context) const {
    const ResourceUsageTag store_tag = tag;
    const ResourceUsageTag transition_tag = tag + 1;
    rp_context.RecordEndRenderPass(external_context, store_tag, transition_tag, env.queue_id);
}

bool EventCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context,
                    ResourceUsageRecord::kMaxIndex, loc);
}

bool EventCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                            ResourceUsageTag replay_tag, const Location& loc) const {
    (void)cb_context;
    const Location command_loc = command == vvl::Func::Empty ? loc : Location(command);
    switch (type) {
        case Type::kSet:
            return ValidateCmdSetEvent(env, events.front(), exec_scope, replay_tag, command_loc);
        case Type::kReset:
            return ValidateCmdResetEvent(env, events.front(), exec_scope, replay_tag, command_loc);
        case Type::kWait: {
            bool skip = ValidateCmdWaitEvents(env, events, replay_tag, command_loc);
            skip |= DetectCmdWaitEventsImageBarrierHazard(env, access_context, events, barrier_sets, replay_tag, command_loc);
            return skip;
        }
    }
    return false;
}

void EventCommand::Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const {
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
            ApplyCmdWaitEvents(env, access_context, events, barrier_sets, tag, command);
            break;
    }
}

}  // namespace syncval
