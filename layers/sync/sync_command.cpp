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
#include "state_tracker/image_state.h"
#include "state_tracker/descriptor_sets.h"
#include "state_tracker/pipeline_state.h"
#include "state_tracker/render_pass_state.h"
#include "sync/sync_command_buffer.h"
#include "sync/sync_image.h"
#include "sync/sync_validation.h"
#include "utils/image_utils.h"

namespace syncval {
namespace {

LogObjectList CommandObjectList(const SyncEnvironment& env, const CommandBufferContext& cb_context,
                                const VulkanTypedHandle& resource) {
    LogObjectList objects(env.handle);
    if (env.queue_id != kQueueIdInvalid) {
        objects.add(cb_context.GetCBState().Handle());
    }
    objects.add(resource);
    return objects;
}

template <typename RegionType>
BufferCopyCommand MakeBufferCopyCommandImpl(std::shared_ptr<const vvl::Buffer> src_buffer,
                                            std::shared_ptr<const vvl::Buffer> dst_buffer, uint32_t region_count,
                                            const RegionType* regions) {
    std::vector<BufferCopyCommand::Region> result;
    result.reserve(region_count);
    for (const auto& region : vvl::make_span(regions, region_count)) {
        const AccessRange src_range = src_buffer ? MakeRange(*src_buffer, region.srcOffset, region.size) : AccessRange{};
        const AccessRange dst_range = dst_buffer ? MakeRange(*dst_buffer, region.dstOffset, region.size) : AccessRange{};
        result.emplace_back(BufferCopyCommand::Region{src_range, dst_range});
    }
    return BufferCopyCommand{std::move(src_buffer), std::move(dst_buffer), std::move(result)};
}

}  // namespace

BufferCopyCommand MakeBufferCopyCommand(std::shared_ptr<const vvl::Buffer> src_buffer,
                                        std::shared_ptr<const vvl::Buffer> dst_buffer, uint32_t region_count,
                                        const VkBufferCopy* regions) {
    return MakeBufferCopyCommandImpl(std::move(src_buffer), std::move(dst_buffer), region_count, regions);
}

BufferCopyCommand MakeBufferCopyCommand(std::shared_ptr<const vvl::Buffer> src_buffer,
                                        std::shared_ptr<const vvl::Buffer> dst_buffer, uint32_t region_count,
                                        const VkBufferCopy2* regions) {
    return MakeBufferCopyCommandImpl(std::move(src_buffer), std::move(dst_buffer), region_count, regions);
}

bool BufferCopyCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool BufferCopyCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                 const CommandBufferContext& cb_context, vvl::Func command,
                                 const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    for (const auto [region_index, region] : vvl::enumerate(regions)) {
        // Includes hazards deduplicated at replay, so replay stops at the same region
        // where record-time validation stopped
        bool region_hazard = false;
        if (src_buffer) {
            const auto hazard = access_context.DetectHazard(*src_buffer, SYNC_COPY_TRANSFER_READ, region.src_range);
            if (hazard.IsHazard()) {
                region_hazard = true;
                if (!record_time_validated_tags.includes(hazard.Tag())) {
                    const LogObjectList objlist = CommandObjectList(env, cb_context, src_buffer->Handle());
                    const std::string error = sync_state.error_messages_.BufferCopyError(
                        hazard, env, command, sync_state.FormatHandle(src_buffer->Handle()), static_cast<uint32_t>(region_index),
                        region.src_range);
                    skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
                }
            }
        }
        if (dst_buffer) {
            const auto hazard = access_context.DetectHazard(*dst_buffer, SYNC_COPY_TRANSFER_WRITE, region.dst_range);
            if (hazard.IsHazard()) {
                region_hazard = true;
                if (!record_time_validated_tags.includes(hazard.Tag())) {
                    const LogObjectList objlist = CommandObjectList(env, cb_context, dst_buffer->Handle());
                    const std::string error = sync_state.error_messages_.BufferCopyError(
                        hazard, env, command, sync_state.FormatHandle(dst_buffer->Handle()), static_cast<uint32_t>(region_index),
                        region.dst_range);
                    skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
                }
            }
        }
        if (region_hazard) {
            break;
        }
    }
    return skip;
}

void BufferCopyCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    if (src_buffer) {
        src_handle_index = cb_context.AddCommandHandle(tag, src_buffer->Handle()).handle_index;
    }
    if (dst_buffer) {
        dst_handle_index = cb_context.AddCommandHandle(tag, dst_buffer->Handle()).handle_index;
    }

    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), tag);
    }
}

void BufferCopyCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    const ResourceUsageTagEx src_tag_ex{tag, src_handle_index};
    const ResourceUsageTagEx dst_tag_ex{tag, dst_handle_index};
    for (const Region& region : regions) {
        if (src_buffer) {
            access_context.UpdateAccessState(*src_buffer, SYNC_COPY_TRANSFER_READ, region.src_range, src_tag_ex, 0, env.queue_id);
        }
        if (dst_buffer) {
            access_context.UpdateAccessState(*dst_buffer, SYNC_COPY_TRANSFER_WRITE, region.dst_range, dst_tag_ex, 0, env.queue_id);
        }
    }
}

BufferAccessCommand MakeBufferAccessCommand(std::shared_ptr<const vvl::Buffer> buffer, SyncAccessIndex access_index,
                                            const AccessRange& range, uint8_t flags, const VulkanTypedHandle& extra_handle) {
    BufferAccessCommand access;
    access.buffer = std::move(buffer);
    access.access_index = access_index;
    access.range = range;
    access.flags = flags;
    access.extra_handle = extra_handle;
    return access;
}

bool BufferAccessCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    const AccessContext& access_context =
        (flags & SyncFlag::kMarker) ? cb_context.GetCurrentAccessContext() : cb_context.GetCbAccessContext();
    return Validate(cb_context.GetSyncEnvironment(), access_context, cb_context, loc.function, ResourceUsageRange{}, loc);
}

bool BufferAccessCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                   const CommandBufferContext& cb_context, vvl::Func command,
                                   const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    if (!buffer) {
        return false;
    }
    const HazardResult hazard = (flags & SyncFlag::kMarker) ? access_context.DetectMarkerHazard(*buffer, range)
                                                            : access_context.DetectHazard(*buffer, access_index, range);
    if (!hazard.IsHazard() || record_time_validated_tags.includes(hazard.Tag())) {
        return false;
    }
    const SyncValidator& sync_state = env.validator;
    LogObjectList objlist;
    if (flags & SyncFlag::kMarker) {
        // The legacy marker report contains only the buffer handle
        objlist.add(buffer->Handle());
    } else {
        objlist.add(env.handle);
        if (env.queue_id != kQueueIdInvalid) {
            objlist.add(cb_context.GetCBState().Handle());
        }
        if (extra_handle.type != kVulkanObjectTypeUnknown) {
            objlist.add(extra_handle);
        }
        objlist.add(buffer->Handle());
    }
    const std::string resource_description = "dstBuffer " + sync_state.FormatHandle(buffer->Handle());
    const std::string error = sync_state.error_messages_.BufferError(hazard, env, command, resource_description, range);
    return sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
}

void BufferAccessCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_access) {
    if (!buffer) {
        return;
    }
    handle_index = cb_context.AddCommandHandle(tag, buffer->Handle()).handle_index;
    if (apply_access) {
        AccessContext& access_context =
            (flags & SyncFlag::kMarker) ? cb_context.GetCurrentAccessContext() : cb_context.GetCbAccessContext();
        Apply(cb_context.GetSyncEnvironment(), access_context, tag);
    }
}

void BufferAccessCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    if (!buffer) {
        return;
    }
    access_context.UpdateAccessState(*buffer, access_index, range, ResourceUsageTagEx{tag, handle_index}, flags, env.queue_id);
}

namespace {

void AddImageAccess(ImageTransferCommand& command, const std::shared_ptr<const vvl::Image>& image, SyncAccessIndex access_index,
                    const VkImageSubresourceLayers& subresource, const VkOffset3D& offset, const VkExtent3D& extent,
                    uint32_t region_index) {
    if (image) {
        command.image_accesses.emplace_back(
            ImageTransferCommand::ImageAccess{image, access_index, RangeFromLayers(subresource), offset, extent, region_index});
    }
}

template <typename RegionType>
ImageTransferCommand MakeImageCopyCommandImpl(const std::shared_ptr<const vvl::Image>& src_image,
                                              const std::shared_ptr<const vvl::Image>& dst_image, uint32_t region_count,
                                              const RegionType* regions, SyncAccessIndex src_access, SyncAccessIndex dst_access) {
    ImageTransferCommand command;
    command.image_accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        AddImageAccess(command, src_image, src_access, region.srcSubresource, region.srcOffset, region.extent, region_index);
        AddImageAccess(command, dst_image, dst_access, region.dstSubresource, region.dstOffset, region.extent, region_index);
    }
    return command;
}

template <typename RegionType>
ImageTransferCommand MakeBufferImageCopyCommandImpl(const std::shared_ptr<const vvl::Buffer>& buffer,
                                                    const std::shared_ptr<const vvl::Image>& image, bool buffer_is_source,
                                                    uint32_t region_count, const RegionType* regions) {
    ImageTransferCommand command;
    if (!image) {
        // Matches the existing behavior: without the image state the buffer side is not validated either
        return command;
    }
    command.buffer_accesses_first = buffer_is_source;
    command.image_accesses.reserve(region_count);
    const SyncAccessIndex buffer_access = buffer_is_source ? SYNC_COPY_TRANSFER_READ : SYNC_COPY_TRANSFER_WRITE;
    const SyncAccessIndex image_access = buffer_is_source ? SYNC_COPY_TRANSFER_WRITE : SYNC_COPY_TRANSFER_READ;
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (buffer) {
            const AccessRange range = MakeRange(region.bufferOffset, image->GetBufferSizeFromCopyImage(region));
            command.buffer_accesses.emplace_back(ImageTransferCommand::BufferAccess{buffer, buffer_access, range, region_index});
        }
        AddImageAccess(command, image, image_access, region.imageSubresource, region.imageOffset, region.imageExtent, region_index);
    }
    return command;
}

template <typename RegionType>
ImageTransferCommand MakeImageBlitCommandImpl(const std::shared_ptr<const vvl::Image>& src_image,
                                              const std::shared_ptr<const vvl::Image>& dst_image, uint32_t region_count,
                                              const RegionType* regions) {
    // Blit offsets can define a flipped region, normalize them to offset + extent form
    auto normalized_offset = [](const auto& offsets) {
        return VkOffset3D{std::min(offsets[0].x, offsets[1].x), std::min(offsets[0].y, offsets[1].y),
                          std::min(offsets[0].z, offsets[1].z)};
    };
    auto normalized_extent = [](const auto& offsets) {
        return VkExtent3D{static_cast<uint32_t>(std::abs(offsets[1].x - offsets[0].x)),
                          static_cast<uint32_t>(std::abs(offsets[1].y - offsets[0].y)),
                          static_cast<uint32_t>(std::abs(offsets[1].z - offsets[0].z))};
    };
    ImageTransferCommand command;
    command.image_accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        AddImageAccess(command, src_image, SYNC_BLIT_TRANSFER_READ, region.srcSubresource, normalized_offset(region.srcOffsets),
                       normalized_extent(region.srcOffsets), region_index);
        AddImageAccess(command, dst_image, SYNC_BLIT_TRANSFER_WRITE, region.dstSubresource, normalized_offset(region.dstOffsets),
                       normalized_extent(region.dstOffsets), region_index);
    }
    return command;
}

}  // namespace

ImageTransferCommand MakeImageCopyCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageCopy* regions) {
    return MakeImageCopyCommandImpl(src_image, dst_image, region_count, regions, SYNC_COPY_TRANSFER_READ, SYNC_COPY_TRANSFER_WRITE);
}

ImageTransferCommand MakeImageCopyCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageCopy2* regions) {
    return MakeImageCopyCommandImpl(src_image, dst_image, region_count, regions, SYNC_COPY_TRANSFER_READ, SYNC_COPY_TRANSFER_WRITE);
}

ImageTransferCommand MakeBufferImageCopyCommand(std::shared_ptr<const vvl::Buffer> buffer, std::shared_ptr<const vvl::Image> image,
                                                bool buffer_is_source, uint32_t region_count, const VkBufferImageCopy* regions) {
    return MakeBufferImageCopyCommandImpl(buffer, image, buffer_is_source, region_count, regions);
}

ImageTransferCommand MakeBufferImageCopyCommand(std::shared_ptr<const vvl::Buffer> buffer, std::shared_ptr<const vvl::Image> image,
                                                bool buffer_is_source, uint32_t region_count, const VkBufferImageCopy2* regions) {
    return MakeBufferImageCopyCommandImpl(buffer, image, buffer_is_source, region_count, regions);
}

ImageTransferCommand MakeImageBlitCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageBlit* regions) {
    return MakeImageBlitCommandImpl(src_image, dst_image, region_count, regions);
}

ImageTransferCommand MakeImageBlitCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageBlit2* regions) {
    return MakeImageBlitCommandImpl(src_image, dst_image, region_count, regions);
}

ImageTransferCommand MakeImageResolveCommand(std::shared_ptr<const vvl::Image> src_image,
                                             std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                             const VkImageResolve* regions) {
    return MakeImageCopyCommandImpl(src_image, dst_image, region_count, regions, SYNC_RESOLVE_TRANSFER_READ,
                                    SYNC_RESOLVE_TRANSFER_WRITE);
}

ImageTransferCommand MakeImageResolveCommand(std::shared_ptr<const vvl::Image> src_image,
                                             std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                             const VkImageResolve2* regions) {
    return MakeImageCopyCommandImpl(src_image, dst_image, region_count, regions, SYNC_RESOLVE_TRANSFER_READ,
                                    SYNC_RESOLVE_TRANSFER_WRITE);
}

ImageTransferCommand MakeImageClearCommand(std::shared_ptr<const vvl::Image> image, uint32_t range_count,
                                           const VkImageSubresourceRange* ranges) {
    ImageTransferCommand command;
    if (!image) {
        return command;
    }
    // The clear commands validate every range even after a hazard is found
    command.stop_at_hazardous_region = false;
    command.image_accesses.reserve(range_count);
    for (const auto [range_index, range] : vvl::enumerate(ranges, range_count)) {
        // Zero extent: the access covers the whole subresource range
        command.image_accesses.emplace_back(
            ImageTransferCommand::ImageAccess{image, SYNC_CLEAR_TRANSFER_WRITE, range, VkOffset3D{}, VkExtent3D{}, range_index});
    }
    return command;
}

bool ImageTransferCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool ImageTransferCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                    const CommandBufferContext& cb_context, vvl::Func command,
                                    const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;
    // Set when the current region has a hazard, including hazards deduplicated at replay,
    // so replay stops at the same region where record-time validation stopped
    bool region_hazard = false;

    auto validate_image_access = [&](const ImageAccess& access) {
        const bool whole_subresource = access.extent.width == 0;
        const HazardResult hazard = whole_subresource
                                        ? access_context.DetectHazard(*access.image, access.subresource_range, access.access_index)
                                        : access_context.DetectHazard(*access.image, access.subresource_range, access.offset,
                                                                      access.extent, access.access_index);
        if (!hazard.IsHazard()) {
            return;
        }
        region_hazard = true;
        if (record_time_validated_tags.includes(hazard.Tag())) {
            return;
        }
        const LogObjectList objlist = CommandObjectList(env, cb_context, access.image->Handle());
        const std::string resource_description = sync_state.FormatHandle(access.image->Handle());
        std::string error;
        if (whole_subresource) {
            error = sync_state.error_messages_.ImageClearError(hazard, env, command, resource_description, access.region_index,
                                                               access.subresource_range);
        } else {
            const VkImageSubresourceLayers subresource{access.subresource_range.aspectMask, access.subresource_range.baseMipLevel,
                                                       access.subresource_range.baseArrayLayer,
                                                       access.subresource_range.layerCount};
            error = sync_state.error_messages_.ImageCopyResolveBlitError(
                hazard, env, command, resource_description, access.region_index, access.offset, access.extent, subresource);
        }
        skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
    };

    auto validate_buffer_access = [&](const BufferAccess& access) {
        const HazardResult hazard = access_context.DetectHazard(*access.buffer, access.access_index, access.range);
        if (!hazard.IsHazard()) {
            return;
        }
        region_hazard = true;
        if (record_time_validated_tags.includes(hazard.Tag())) {
            return;
        }
        const LogObjectList objlist = CommandObjectList(env, cb_context, access.buffer->Handle());
        const std::string error = sync_state.error_messages_.BufferCopyError(
            hazard, env, command, sync_state.FormatHandle(access.buffer->Handle()), access.region_index, access.range);
        skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
    };

    // Region by region, source accesses before destination accesses, stopping after the
    // first region with a hazard - matches the legacy per-command validation loops
    size_t image_i = 0;
    size_t buffer_i = 0;
    while (image_i < image_accesses.size() || buffer_i < buffer_accesses.size()) {
        uint32_t region = vvl::kU32Max;
        if (image_i < image_accesses.size()) {
            region = std::min(region, image_accesses[image_i].region_index);
        }
        if (buffer_i < buffer_accesses.size()) {
            region = std::min(region, buffer_accesses[buffer_i].region_index);
        }
        region_hazard = false;
        if (buffer_accesses_first) {
            while (buffer_i < buffer_accesses.size() && buffer_accesses[buffer_i].region_index == region) {
                validate_buffer_access(buffer_accesses[buffer_i++]);
            }
        }
        while (image_i < image_accesses.size() && image_accesses[image_i].region_index == region) {
            validate_image_access(image_accesses[image_i++]);
        }
        if (!buffer_accesses_first) {
            while (buffer_i < buffer_accesses.size() && buffer_accesses[buffer_i].region_index == region) {
                validate_buffer_access(buffer_accesses[buffer_i++]);
            }
        }
        if (region_hazard && stop_at_hazardous_region) {
            break;
        }
    }
    return skip;
}

void ImageTransferCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    // One handle record per distinct resource
    for (size_t i = 0; i < image_accesses.size(); i++) {
        uint32_t handle_index = vvl::kNoIndex32;
        for (size_t previous = 0; previous < i; previous++) {
            if (image_accesses[previous].image == image_accesses[i].image) {
                handle_index = image_accesses[previous].handle_index;
                break;
            }
        }
        if (handle_index == vvl::kNoIndex32) {
            handle_index = cb_context.AddCommandHandle(tag, image_accesses[i].image->Handle()).handle_index;
        }
        image_accesses[i].handle_index = handle_index;
    }
    for (size_t i = 0; i < buffer_accesses.size(); i++) {
        uint32_t handle_index = vvl::kNoIndex32;
        for (size_t previous = 0; previous < i; previous++) {
            if (buffer_accesses[previous].buffer == buffer_accesses[i].buffer) {
                handle_index = buffer_accesses[previous].handle_index;
                break;
            }
        }
        if (handle_index == vvl::kNoIndex32) {
            handle_index = cb_context.AddCommandHandle(tag, buffer_accesses[i].buffer->Handle()).handle_index;
        }
        buffer_accesses[i].handle_index = handle_index;
    }
    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCbAccessContext(), tag);
    }
}

void ImageTransferCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    for (const ImageAccess& access : image_accesses) {
        const auto& sub_state = SubState(*access.image);
        ImageRangeGen range_gen = (access.extent.width == 0)
                                      ? sub_state.MakeImageRangeGen(access.subresource_range, false)
                                      : sub_state.MakeImageRangeGen(access.subresource_range, access.offset, access.extent, false);
        access_context.UpdateAccessState(range_gen, access.access_index, ResourceUsageTagEx{tag, access.handle_index}, 0,
                                         env.queue_id);
    }
    for (const BufferAccess& access : buffer_accesses) {
        access_context.UpdateAccessState(*access.buffer, access.access_index, access.range,
                                         ResourceUsageTagEx{tag, access.handle_index}, 0, env.queue_id);
    }
}

bool PipelineBarrierCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool PipelineBarrierCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                      const CommandBufferContext& cb_context, vvl::Func command,
                                      const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
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

namespace {

bool ValidateDescriptorAccesses(const std::vector<DescriptorAccess>& accesses, const vvl::Pipeline& pipeline,
                                const std::vector<std::shared_ptr<const vvl::DescriptorSet>>& descriptor_sets,
                                const SyncEnvironment& env, const AccessContext& access_context,
                                const CommandBufferContext& cb_context, vvl::Func command,
                                const ResourceUsageRange& record_time_validated_tags, const Location& loc) {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    for (const DescriptorAccess& access : accesses) {
        HazardResult hazard;
        if (access.image_view) {
            hazard = access.input_attachment
                         ? access_context.DetectAttachmentHazard(*access.image_view, access.render_offset, access.render_extent,
                                                                 access.access_index, access.attachment_access, env.queue_id)
                         : access_context.DetectHazard(*access.image_view, access.access_index);
        } else {
            hazard = access_context.DetectHazard(*access.buffer, access.access_index, access.range);
        }
        if (!hazard.IsHazard() || record_time_validated_tags.includes(hazard.Tag())) {
            continue;
        }
        LogObjectList objlist =
            CommandObjectList(env, cb_context,
                              access.descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR ? access.buffer->Handle()
                                                                                                      : access.description_handle);
        objlist.add(pipeline.Handle());
        const std::string resource_description = sync_state.FormatHandle(access.description_handle);
        const vvl::DescriptorSet& descriptor_set = *descriptor_sets[access.set_index];
        std::string error;
        if (access.image_view) {
            error = sync_state.error_messages_.ImageDescriptorError(hazard, env, command, resource_description, pipeline,
                                                                    access.set_number, descriptor_set, access.descriptor_type,
                                                                    access.binding, 0, access.stage, access.image_layout);
        } else if (access.descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
            error = sync_state.error_messages_.AccelerationStructureDescriptorError(
                hazard, env, command, resource_description, pipeline, access.set_number, descriptor_set, access.descriptor_type,
                access.binding, 0, access.stage);
        } else {
            error = sync_state.error_messages_.BufferDescriptorError(hazard, env, command, resource_description, pipeline,
                                                                     access.set_number, descriptor_set, access.descriptor_type,
                                                                     access.binding, 0, access.stage);
        }
        skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
    }
    return skip;
}

void RecordDescriptorAccessHandles(std::vector<DescriptorAccess>& accesses, CommandBufferContext& cb_context,
                                   ResourceUsageTag tag) {
    for (DescriptorAccess& access : accesses) {
        access.handle_index = cb_context.AddCommandHandle(tag, access.command_handle).handle_index;
    }
}

void ApplyDescriptorAccesses(const std::vector<DescriptorAccess>& accesses, SyncEnvironment& env, AccessContext& access_context,
                             ResourceUsageTag tag) {
    for (const DescriptorAccess& access : accesses) {
        const ResourceUsageTagEx tag_ex{tag, access.handle_index};
        if (access.image_view) {
            if (access.input_attachment) {
                ImageRangeGen range_gen(MakeImageRangeGen(*access.image_view, access.render_offset, access.render_extent));
                access_context.UpdateAttachmentAccessState(range_gen, access.access_index, access.attachment_access, tag_ex,
                                                           env.queue_id);
            } else {
                ImageRangeGen range_gen = MakeImageRangeGen(*access.image_view);
                access_context.UpdateAccessState(range_gen, access.access_index, tag_ex, 0, env.queue_id);
            }
        } else {
            access_context.UpdateAccessState(*access.buffer, access.access_index, access.range, tag_ex, 0, env.queue_id);
        }
    }
}

}  // namespace

DispatchCommand MakeDispatchCommand(const CommandBufferContext& cb_context, VkPipelineBindPoint bind_point,
                                    std::shared_ptr<const vvl::Buffer> indirect_buffer, VkDeviceSize indirect_offset) {
    DispatchCommand command;
    cb_context.CollectDescriptorAccesses(bind_point, command.descriptor_accesses, command.pipeline, command.descriptor_sets);
    if (indirect_buffer) {
        command.indirect_range = MakeRange(indirect_offset, sizeof(VkDispatchIndirectCommand));
        command.indirect_buffer = std::move(indirect_buffer);
    }
    return command;
}

bool DispatchCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool DispatchCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                               const CommandBufferContext& cb_context, vvl::Func command,
                               const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    if (pipeline) {
        skip |= ValidateDescriptorAccesses(descriptor_accesses, *pipeline, descriptor_sets, env, access_context, cb_context,
                                           command, record_time_validated_tags, loc);
    }

    if (indirect_buffer) {
        const HazardResult hazard =
            access_context.DetectHazard(*indirect_buffer, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, indirect_range);
        if (hazard.IsHazard() && !record_time_validated_tags.includes(hazard.Tag())) {
            const LogObjectList objlist = CommandObjectList(env, cb_context, indirect_buffer->Handle());
            const std::string resource_description = "indirect " + sync_state.FormatHandle(indirect_buffer->Handle());
            const std::string error =
                sync_state.error_messages_.BufferError(hazard, env, command, resource_description, indirect_range);
            skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
        }
    }
    return skip;
}

void DispatchCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    RecordDescriptorAccessHandles(descriptor_accesses, cb_context, tag);
    if (indirect_buffer) {
        indirect_handle_index = cb_context.AddCommandHandle(tag, indirect_buffer->Handle()).handle_index;
    }
    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), tag);
    }
}

void DispatchCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    ApplyDescriptorAccesses(descriptor_accesses, env, access_context, tag);
    if (indirect_buffer) {
        access_context.UpdateAccessState(*indirect_buffer, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, indirect_range,
                                         ResourceUsageTagEx{tag, indirect_handle_index}, 0, env.queue_id);
    }
}

DrawCommand MakeDrawCommand(const CommandBufferContext& cb_context) {
    DrawCommand command;
    cb_context.CollectDescriptorAccesses(VK_PIPELINE_BIND_POINT_GRAPHICS, command.descriptor_accesses, command.pipeline,
                                         command.descriptor_sets);
    cb_context.CollectDrawAttachmentAccesses(command.attachment_accesses);
    return command;
}

void AddDrawIndirectAccess(DrawCommand& command, std::shared_ptr<const vvl::Buffer> buffer, VkDeviceSize offset,
                           VkDeviceSize struct_size, uint32_t draw_count, uint32_t stride) {
    if (!buffer || draw_count == 0) {
        return;
    }
    if (draw_count == 1 || stride == struct_size) {
        command.indirect_ranges.emplace_back(MakeRange(offset, struct_size * draw_count));
    } else {
        for (uint32_t i = 0; i < draw_count; ++i) {
            command.indirect_ranges.emplace_back(MakeRange(offset + VkDeviceSize(i) * stride, struct_size));
        }
    }
    command.indirect_buffer = std::move(buffer);
}

void AddDrawCountAccess(DrawCommand& command, std::shared_ptr<const vvl::Buffer> buffer, VkDeviceSize offset,
                        const char* count_label) {
    if (!buffer) {
        return;
    }
    command.count_range = MakeRange(offset, 4);
    command.count_buffer = std::move(buffer);
    command.count_label = count_label;
}

bool DrawCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool DrawCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                           vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    if (pipeline) {
        skip |= ValidateDescriptorAccesses(descriptor_accesses, *pipeline, descriptor_sets, env, access_context, cb_context,
                                           command, record_time_validated_tags, loc);
    }

    // The pipeline handle is part of the report only for the pipeline-defined accesses
    // (vertex and index buffers), matching the legacy per-command report sites
    auto validate_buffer_access = [&](const std::shared_ptr<const vvl::Buffer>& buffer, SyncAccessIndex access_index,
                                      const AccessRange& range, const char* what, bool report_pipeline) {
        const HazardResult hazard = access_context.DetectHazard(*buffer, access_index, range);
        if (!hazard.IsHazard()) {
            return false;
        }
        if (record_time_validated_tags.includes(hazard.Tag())) {
            // Deduplicated at replay, but still a hazard for the early-out decisions below
            return true;
        }
        LogObjectList objlist = CommandObjectList(env, cb_context, buffer->Handle());
        if (report_pipeline && pipeline) {
            objlist.add(pipeline->Handle());
        }
        const std::string resource_description = std::string(what) + " " + sync_state.FormatHandle(buffer->Handle());
        const std::string error = sync_state.error_messages_.BufferError(hazard, env, command, resource_description, range);
        skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
        return true;
    };

    for (const VertexAccess& access : vertex_accesses) {
        validate_buffer_access(access.buffer, SYNC_VERTEX_ATTRIBUTE_INPUT_VERTEX_ATTRIBUTE_READ, access.range, "vertex", true);
    }
    for (const VertexAccess& access : index_accesses) {
        validate_buffer_access(access.buffer, SYNC_INDEX_INPUT_INDEX_READ, access.range, "index", true);
    }

    for (const DrawAttachmentAccess& access : attachment_accesses) {
        ImageRangeGen range_gen = access.range_gen;
        const HazardResult hazard =
            access_context.DetectAttachmentHazard(range_gen, access.access_index, access.attachment_access, env.queue_id);
        if (!hazard.IsHazard() || record_time_validated_tags.includes(hazard.Tag())) {
            continue;
        }
        LogObjectList objlist = CommandObjectList(env, cb_context, access.view->Handle());
        if (access.dynamic_rendering) {
            const std::string error = sync_state.error_messages_.Error(
                env, hazard, command, sync_state.FormatHandle(access.view->Handle()), "DynamicRenderingAttachmentError");
            const Location attachment_loc = (access.attachment_field == vvl::Field::pColorAttachments)
                                                ? loc.dot(vvl::Struct::VkRenderingAttachmentInfo, access.attachment_field,
                                                          access.attachment_index)
                                                : loc.dot(vvl::Struct::VkRenderingAttachmentInfo, access.attachment_field);
            skip |= sync_state.SyncError(hazard.Hazard(), objlist, attachment_loc.dot(vvl::Field::imageView), error);
        } else {
            objlist.add(access.view->image_state->Handle());
            std::ostringstream ss;
            ss << access.description;
            ss << " (" << sync_state.FormatHandle(access.view->Handle());
            ss << ", " << sync_state.FormatHandle(access.view->image_state->Handle()) << ")";
            const std::string error = sync_state.error_messages_.RenderPassAttachmentError(hazard, env, command, ss.str());
            skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
        }
    }

    if (indirect_buffer) {
        for (const AccessRange& range : indirect_ranges) {
            // Matches the legacy multi-draw validation: stop after the first reported range
            if (validate_buffer_access(indirect_buffer, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, range, "indirect", false)) {
                break;
            }
        }
    }
    if (count_buffer) {
        validate_buffer_access(count_buffer, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, count_range, count_label, false);
    }
    return skip;
}

void DrawCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    RecordDescriptorAccessHandles(descriptor_accesses, cb_context, tag);
    for (VertexAccess& access : vertex_accesses) {
        access.handle_index = cb_context.AddCommandHandle(tag, access.buffer->Handle()).handle_index;
    }
    for (VertexAccess& access : index_accesses) {
        access.handle_index = cb_context.AddCommandHandle(tag, access.buffer->Handle()).handle_index;
    }
    if (indirect_buffer) {
        indirect_handle_index = cb_context.AddCommandHandle(tag, indirect_buffer->Handle()).handle_index;
    }
    if (count_buffer) {
        count_handle_index = cb_context.AddCommandHandle(tag, count_buffer->Handle()).handle_index;
    }
    // The attachment accesses do not register handle records (matches the previous behavior)
    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), tag);
    }
}

void DrawCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    ApplyDescriptorAccesses(descriptor_accesses, env, access_context, tag);
    for (const VertexAccess& access : vertex_accesses) {
        access_context.UpdateAccessState(*access.buffer, SYNC_VERTEX_ATTRIBUTE_INPUT_VERTEX_ATTRIBUTE_READ, access.range,
                                         ResourceUsageTagEx{tag, access.handle_index}, 0, env.queue_id);
    }
    for (const VertexAccess& access : index_accesses) {
        access_context.UpdateAccessState(*access.buffer, SYNC_INDEX_INPUT_INDEX_READ, access.range,
                                         ResourceUsageTagEx{tag, access.handle_index}, 0, env.queue_id);
    }
    if (indirect_buffer) {
        for (const AccessRange& range : indirect_ranges) {
            access_context.UpdateAccessState(*indirect_buffer, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, range,
                                             ResourceUsageTagEx{tag, indirect_handle_index}, 0, env.queue_id);
        }
    }
    if (count_buffer) {
        access_context.UpdateAccessState(*count_buffer, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, count_range,
                                         ResourceUsageTagEx{tag, count_handle_index}, 0, env.queue_id);
    }
    for (const DrawAttachmentAccess& access : attachment_accesses) {
        ImageRangeGen range_gen = access.range_gen;
        access_context.UpdateAttachmentAccessState(range_gen, access.access_index, access.attachment_access,
                                                   ResourceUsageTagEx{tag}, env.queue_id);
    }
}

bool EventCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context, ResourceUsageTag tag,
                            const ResourceUsageRange& record_time_validated_tags) const {
    if (const auto* set_event = std::get_if<SetEventReplay>(&operation)) {
        return ValidateCmdSetEvent(env, set_event->event, set_event->src_exec_scope, tag, Location(set_event->command),
                                   record_time_validated_tags);
    }
    if (const auto* reset_event = std::get_if<ResetEventReplay>(&operation)) {
        return ValidateCmdResetEvent(env, reset_event->event, reset_event->exec_scope, tag, Location(reset_event->command),
                                     record_time_validated_tags);
    }
    const auto* wait_events = std::get_if<WaitEventsReplay>(&operation);
    assert(wait_events);
    bool skip = false;
    const Location loc(wait_events->command);
    skip |= ValidateCmdWaitEvents(env, wait_events->events, tag, loc, record_time_validated_tags);
    skip |= DetectCmdWaitEventsImageBarrierHazard(env, access_context, wait_events->events, wait_events->barrier_sets, tag, loc,
                                                  record_time_validated_tags);
    return skip;
}

void EventCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    if (const auto* set_event = std::get_if<SetEventReplay>(&operation)) {
        // The replay has already applied every command preceding the set event, so the event's
        // first scope is exactly the current state of the replay context. The record time
        // snapshot is only for the legacy first access replay: resolving it here would add the
        // same accesses again, with tags offset by the set event's position instead of the
        // submission base.
        auto src_access_context = std::make_shared<AccessContext>(*access_context.validator);
        src_access_context->InitFrom(access_context);
        src_access_context->TrimAndClearFirstAccess();

        ApplyCmdSetEvent(env, set_event->event, set_event->src_exec_scope, src_access_context, tag, set_event->command);
    } else if (const auto* reset_event = std::get_if<ResetEventReplay>(&operation)) {
        ApplyCmdResetEvent(env, reset_event->event, tag, reset_event->command);
    } else if (const auto* wait_events = std::get_if<WaitEventsReplay>(&operation)) {
        // Command replay owns the complete queue context, so the layout transition writes
        // of the wait's image barriers must be materialized (the legacy first access replay
        // resolves them from the command buffer access summary instead).
        ApplyCmdWaitEvents(env, access_context, wait_events->events, wait_events->barrier_sets, tag, wait_events->command, true);
    }
}

TraceRaysCommand MakeTraceRaysCommand(const CommandBufferContext& cb_context) {
    TraceRaysCommand command;
    cb_context.CollectDescriptorAccesses(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, command.descriptor_accesses, command.pipeline,
                                         command.descriptor_sets);
    return command;
}

bool TraceRaysCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool TraceRaysCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                const CommandBufferContext& cb_context, vvl::Func command,
                                const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    if (pipeline) {
        skip |= ValidateDescriptorAccesses(descriptor_accesses, *pipeline, descriptor_sets, env, access_context, cb_context,
                                           command, record_time_validated_tags, loc);
    }

    for (const SbtAccess& sbt : sbt_accesses) {
        const HazardResult hazard =
            access_context.DetectHazard(*sbt.buffer, SYNC_RAY_TRACING_SHADER_SHADER_BINDING_TABLE_READ, sbt.range);
        if (hazard.IsHazard() && !record_time_validated_tags.includes(hazard.Tag())) {
            const LogObjectList objlist = CommandObjectList(env, cb_context, sbt.buffer->Handle());
            const std::string resource_description =
                std::string(sbt.label) + " shader binding table " + sync_state.FormatHandle(sbt.buffer->Handle());
            const std::string error = sync_state.error_messages_.BufferError(hazard, env, command, resource_description, sbt.range);
            skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
        }
    }

    if (indirect_buffer) {
        const HazardResult hazard =
            access_context.DetectHazard(*indirect_buffer, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, indirect_range);
        if (hazard.IsHazard() && !record_time_validated_tags.includes(hazard.Tag())) {
            const LogObjectList objlist = CommandObjectList(env, cb_context, indirect_buffer->Handle());
            const std::string resource_description = "indirect " + sync_state.FormatHandle(indirect_buffer->Handle());
            const std::string error =
                sync_state.error_messages_.BufferError(hazard, env, command, resource_description, indirect_range);
            skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
        }
    }
    return skip;
}

void TraceRaysCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    RecordDescriptorAccessHandles(descriptor_accesses, cb_context, tag);
    for (SbtAccess& sbt : sbt_accesses) {
        sbt.handle_index = cb_context.AddCommandHandle(tag, sbt.buffer->Handle()).handle_index;
    }
    if (indirect_buffer) {
        indirect_handle_index = cb_context.AddCommandHandle(tag, indirect_buffer->Handle()).handle_index;
    }
    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), tag);
    }
}

void TraceRaysCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    ApplyDescriptorAccesses(descriptor_accesses, env, access_context, tag);
    for (const SbtAccess& sbt : sbt_accesses) {
        access_context.UpdateAccessState(*sbt.buffer, SYNC_RAY_TRACING_SHADER_SHADER_BINDING_TABLE_READ, sbt.range,
                                         ResourceUsageTagEx{tag, sbt.handle_index}, 0, env.queue_id);
    }
    if (indirect_buffer) {
        access_context.UpdateAccessState(*indirect_buffer, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, indirect_range,
                                         ResourceUsageTagEx{tag, indirect_handle_index}, 0, env.queue_id);
    }
}

bool AccelerationStructureCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool AccelerationStructureCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                            const CommandBufferContext& cb_context, vvl::Func command,
                                            const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    for (const BufferAccess& access : accesses) {
        const HazardResult hazard = access_context.DetectHazard(*access.buffer, access.access_index, access.range);
        if (!hazard.IsHazard() || record_time_validated_tags.includes(hazard.Tag())) {
            continue;
        }
        LogObjectList objlist = CommandObjectList(env, cb_context, access.buffer->Handle());
        std::string error;
        if (access.acceleration_structure != VK_NULL_HANDLE) {
            objlist.add(access.acceleration_structure);
            const std::string resource_description = sync_state.FormatHandle(access.buffer->VkHandle());
            error = sync_state.error_messages_.AccelerationStructureError(hazard, env, command, resource_description, access.range,
                                                                          access.acceleration_structure, access.description);
        } else {
            error = sync_state.error_messages_.BufferError(hazard, env, command, access.description, access.range);
        }
        skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
    }
    return skip;
}

void AccelerationStructureCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    for (BufferAccess& access : accesses) {
        if (access.apply) {
            access.handle_index = cb_context.AddCommandHandle(tag, access.buffer->Handle()).handle_index;
        }
    }
    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), tag);
    }
}

void AccelerationStructureCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    for (const BufferAccess& access : accesses) {
        if (!access.apply) {
            continue;
        }
        access_context.UpdateAccessState(*access.buffer, access.access_index, access.range,
                                         ResourceUsageTagEx{tag, access.handle_index}, 0, env.queue_id);
    }
}

bool VideoCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool VideoCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                            vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    if (bitstream_buffer) {
        const HazardResult hazard = access_context.DetectHazard(*bitstream_buffer, bitstream_access_index, bitstream_range);
        if (hazard.IsHazard() && !record_time_validated_tags.includes(hazard.Tag())) {
            const std::string error =
                sync_state.error_messages_.BufferError(hazard, env, command, bitstream_description, bitstream_range);
            skip |= sync_state.SyncError(hazard.Hazard(), bitstream_buffer->Handle(), loc, error);
        }
    }
    for (const PictureAccess& picture : picture_accesses) {
        ImageRangeGen range_gen = picture.range_gen;  // detection consumes the generator, use a copy
        const HazardResult hazard = access_context.DetectHazard(range_gen, picture.access_index);
        if (hazard.IsHazard() && !record_time_validated_tags.includes(hazard.Tag())) {
            const std::string error = sync_state.error_messages_.VideoError(hazard, env, command, picture.description);
            skip |= sync_state.SyncError(hazard.Hazard(), picture.view->Handle(), loc, error);
        }
    }
    return skip;
}

void VideoCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    if (bitstream_buffer) {
        bitstream_handle_index = cb_context.AddCommandHandle(tag, bitstream_buffer->Handle()).handle_index;
    }
    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), tag);
    }
}

void VideoCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    if (bitstream_buffer) {
        access_context.UpdateAccessState(*bitstream_buffer, bitstream_access_index, bitstream_range,
                                         ResourceUsageTagEx{tag, bitstream_handle_index}, 0, env.queue_id);
    }
    for (const PictureAccess& picture : picture_accesses) {
        ImageRangeGen range_gen = picture.range_gen;
        access_context.UpdateAccessState(range_gen, picture.access_index, ResourceUsageTagEx{tag}, 0, env.queue_id);
    }
}

bool ClearAttachmentsCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool ClearAttachmentsCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                       const CommandBufferContext& cb_context, vvl::Func command,
                                       const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    for (const ClearAccess& access : accesses) {
        ImageRangeGen range_gen = access.range_gen;  // detection consumes the generator, use a copy
        const HazardResult hazard =
            access_context.DetectAttachmentHazard(range_gen, access.access_index, access.attachment_access, env.queue_id);
        if (hazard.IsHazard() && !record_time_validated_tags.includes(hazard.Tag())) {
            const LogObjectList objlist = CommandObjectList(env, cb_context, access.view->Handle());
            const std::string error = sync_state.error_messages_.ClearAttachmentError(
                hazard, env, command, access.description, access.aspect_mask, access.rect_index, access.rect);
            skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
        }
    }
    return skip;
}

void ClearAttachmentsCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), tag);
    }
}

void ClearAttachmentsCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    for (const ClearAccess& access : accesses) {
        ImageRangeGen range_gen = access.range_gen;
        access_context.UpdateAttachmentAccessState(range_gen, access.access_index, access.attachment_access,
                                                   ResourceUsageTagEx{tag}, env.queue_id);
    }
}

bool DynamicRenderingCommand::Validate(const CommandBufferContext& cb_context, const Location& loc) const {
    return Validate(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), cb_context, loc.function,
                    ResourceUsageRange{}, loc);
}

bool DynamicRenderingCommand::Validate(const SyncEnvironment& env, const AccessContext& access_context,
                                       const CommandBufferContext& cb_context, vvl::Func command,
                                       const ResourceUsageRange& record_time_validated_tags, const Location& loc) const {
    bool skip = false;
    const SyncValidator& sync_state = env.validator;

    for (const Access& access : accesses) {
        ImageRangeGen range_gen = access.range_gen;  // detection consumes the generator, use a copy
        const HazardResult hazard =
            access_context.DetectAttachmentHazard(range_gen, access.access_index, access.attachment_access, env.queue_id);
        if (!hazard.IsHazard() || record_time_validated_tags.includes(hazard.Tag())) {
            continue;
        }
        const LogObjectList objlist = CommandObjectList(env, cb_context, access.view->Handle());
        std::string error;
        switch (access.op_type) {
            case OpType::kLoad:
                error = sync_state.error_messages_.BeginRenderingError(hazard, env, command, access.description,
                                                                       VkAttachmentLoadOp(access.op));
                break;
            case OpType::kResolveRead:
            case OpType::kResolveWrite:
                error = sync_state.error_messages_.EndRenderingResolveError(hazard, env, command, access.description,
                                                                            VkResolveModeFlagBits(access.op),
                                                                            access.op_type == OpType::kResolveWrite);
                break;
            case OpType::kStore:
                error = sync_state.error_messages_.EndRenderingStoreError(hazard, env, command, access.description,
                                                                          VkAttachmentStoreOp(access.op));
                break;
        }
        skip |= sync_state.SyncError(hazard.Hazard(), objlist, loc, error);
        if (skip) {
            break;
        }
    }
    return skip;
}

void DynamicRenderingCommand::Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses) {
    if (apply_accesses) {
        Apply(cb_context.GetSyncEnvironment(), cb_context.GetCurrentAccessContext(), tag);
    }
}

void DynamicRenderingCommand::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const {
    for (const Access& access : accesses) {
        ImageRangeGen range_gen = access.range_gen;
        access_context.UpdateAttachmentAccessState(range_gen, access.access_index, access.attachment_access,
                                                   ResourceUsageTagEx{tag}, env.queue_id);
    }
}

bool ReplayRecordedCommands(SyncEnvironment& env, AccessContext& access_context, const CommandBufferContext& cb_context,
                            ResourceUsageTag base_tag, const ResourceUsageRange& record_time_validated_tags, const Location& loc) {
    bool skip = false;
    // Inside a render pass instance the commands are replayed against the reconstructed
    // subpass context; the render pass structural commands switch it.
    std::unique_ptr<RenderPassAccessContext> replay_rp_context;
    AccessContext* current_context = &access_context;
    for (const RecordedCommandEntry& entry : cb_context.GetRecordedCommands()) {
        const ResourceUsageTag replay_tag = base_tag + entry.tag;
        if (const auto* event_command = std::get_if<EventCommand>(&entry.command)) {
            // Event validation additionally filters by the replay tag: event state reflects the
            // preceding commands of the current replay, unlike access state tag ranges
            skip |= event_command->Validate(env, *current_context, replay_tag, record_time_validated_tags);
            event_command->Apply(env, *current_context, replay_tag);
            continue;
        }
        const vvl::Func command = cb_context.GetResourceUsageInfo(ResourceUsageTagEx{entry.tag}).command;
        if (const auto* begin_render_pass = std::get_if<BeginRenderPassCommand>(&entry.command)) {
            const vvl::RenderPass& rp_state = *begin_render_pass->rp_state;
            const uint32_t subpass_zero = 0;
            const uint32_t view_mask = rp_state.create_info.pSubpasses[0].viewMask;
            const uint32_t instance_id = begin_render_pass->render_pass_instance_id;
            const AttachmentViewGenVector view_gens = RenderPassAccessContext::CreateAttachmentViewGen(
                begin_render_pass->render_area, begin_render_pass->attachment_views);

            // Validate the subpass 0 layout transitions and load operations the same way the
            // record time validation does: against a temporary subpass context.
            AccessContext temp_context(env.validator);
            temp_context.InitFrom(subpass_zero, env.queue_flags, rp_state.subpass_dependency_infos, nullptr, *current_context);
            bool rp_skip = RenderPassAccessContext::ValidateLayoutTransitions(env, record_time_validated_tags, cb_context,
                                                                              temp_context, rp_state, instance_id, subpass_zero,
                                                                              view_mask, view_gens, command);
            if (!rp_skip) {
                RenderPassAccessContext::RecordLayoutTransitions(rp_state, subpass_zero, view_gens, kInvalidTag, temp_context,
                                                                 env.queue_id);
                rp_skip |= RenderPassAccessContext::ValidateLoadOperation(env, record_time_validated_tags, cb_context, temp_context,
                                                                          rp_state, instance_id, subpass_zero, view_mask, view_gens,
                                                                          command);
            }
            skip |= rp_skip;

            replay_rp_context =
                std::make_unique<RenderPassAccessContext>(rp_state, begin_render_pass->render_area, env.queue_flags,
                                                          begin_render_pass->attachment_views, *current_context, instance_id);
            replay_rp_context->RecordBeginRenderPass(replay_tag, replay_tag + 1, env.queue_id);
            current_context = &replay_rp_context->CurrentContext();
            continue;
        }
        if (std::get_if<NextSubpassCommand>(&entry.command)) {
            skip |= replay_rp_context->ValidateNextSubpass(env, record_time_validated_tags, cb_context, command);
            replay_rp_context->RecordNextSubpass(replay_tag, replay_tag + 1, replay_tag + 2, replay_tag + 3, env.queue_id);
            current_context = &replay_rp_context->CurrentContext();
            continue;
        }
        if (std::get_if<EndRenderPassCommand>(&entry.command)) {
            skip |= replay_rp_context->ValidateEndRenderPass(env, record_time_validated_tags, cb_context, command);
            replay_rp_context->RecordEndRenderPass(&access_context, replay_tag, replay_tag + 1, env.queue_id);
            replay_rp_context.reset();
            current_context = &access_context;
            continue;
        }
        if (const auto* execute_commands = std::get_if<ExecuteCommandsCommand>(&entry.command)) {
            // The secondary's log follows the index tag in the primary's tag space
            const CommandBufferContext& secondary_context = GetCommandBufferContext(*execute_commands->secondary_cb);
            skip |=
                ReplayRecordedCommands(env, *current_context, secondary_context, replay_tag + 1, record_time_validated_tags, loc);
            continue;
        }
        if (const auto* buffer_copy = std::get_if<BufferCopyCommand>(&entry.command)) {
            skip |= buffer_copy->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            buffer_copy->Apply(env, *current_context, replay_tag);
        } else if (const auto* buffer_access = std::get_if<BufferAccessCommand>(&entry.command)) {
            skip |= buffer_access->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            buffer_access->Apply(env, *current_context, replay_tag);
        } else if (const auto* image_transfer = std::get_if<ImageTransferCommand>(&entry.command)) {
            skip |= image_transfer->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            image_transfer->Apply(env, *current_context, replay_tag);
        } else if (const auto* dispatch = std::get_if<DispatchCommand>(&entry.command)) {
            skip |= dispatch->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            dispatch->Apply(env, *current_context, replay_tag);
        } else if (const auto* draw = std::get_if<DrawCommand>(&entry.command)) {
            skip |= draw->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            draw->Apply(env, *current_context, replay_tag);
        } else if (const auto* trace_rays = std::get_if<TraceRaysCommand>(&entry.command)) {
            skip |= trace_rays->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            trace_rays->Apply(env, *current_context, replay_tag);
        } else if (const auto* accel = std::get_if<AccelerationStructureCommand>(&entry.command)) {
            skip |= accel->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            accel->Apply(env, *current_context, replay_tag);
        } else if (const auto* video = std::get_if<VideoCommand>(&entry.command)) {
            skip |= video->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            video->Apply(env, *current_context, replay_tag);
        } else if (const auto* clear_attachments = std::get_if<ClearAttachmentsCommand>(&entry.command)) {
            skip |= clear_attachments->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            clear_attachments->Apply(env, *current_context, replay_tag);
        } else if (const auto* dynamic_rendering = std::get_if<DynamicRenderingCommand>(&entry.command)) {
            skip |= dynamic_rendering->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            dynamic_rendering->Apply(env, *current_context, replay_tag);
        } else if (const auto* pipeline_barrier = std::get_if<PipelineBarrierCommand>(&entry.command)) {
            skip |= pipeline_barrier->Validate(env, *current_context, cb_context, command, record_time_validated_tags, loc);
            pipeline_barrier->Apply(env, *current_context, replay_tag);
        }
    }
    return skip;
}

}  // namespace syncval
