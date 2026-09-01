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
#include "sync/sync_image.h"
#include "sync/sync_render_pass.h"
#include "sync/sync_validation.h"
#include "state_tracker/buffer_state.h"
#include "state_tracker/image_state.h"
#include "error_message/logging.h"
#include "utils/image_utils.h"

namespace syncval {

class CommandReplayContext {
  public:
    CommandReplayContext(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag base_tag)
        : env_(env), access_context_(access_context), render_pass_instance_offset_(uint32_t(base_tag)) {}

    AccessContext& CurrentAccessContext() {
        return render_pass_context_ ? render_pass_context_->CurrentContext() : access_context_;
    }

    RenderPassAccessContext* BeginRenderPass(const RenderPassCommand& command) {
        if (!command.render_pass) {
            return nullptr;
        }
        render_pass_context_ = std::make_unique<RenderPassAccessContext>(
            *command.render_pass, command.render_area, env_.queue_flags, command.attachments, access_context_,
            command.render_pass_instance_id + render_pass_instance_offset_);
        return render_pass_context_.get();
    }

    RenderPassAccessContext* GetRenderPassContext() { return render_pass_context_.get(); }
    void EndRenderPass() { render_pass_context_.reset(); }
    AccessContext& ExternalAccessContext() { return access_context_; }
    SyncEnvironment& GetSyncEnvironment() { return env_; }

  private:
    SyncEnvironment& env_;
    AccessContext& access_context_;
    uint32_t render_pass_instance_offset_;
    std::unique_ptr<RenderPassAccessContext> render_pass_context_;
};

static LogObjectList BaseObjectList(const SyncEnvironment& env, const CommandBufferContext& cb_context,
                                    const VulkanTypedHandle& resource, const VulkanTypedHandle& related_object = {}) {
    LogObjectList objlist;
    const VulkanTypedHandle& cb_handle = cb_context.GetCBState().Handle();

    // During recording, env.handle is the command buffer handle. Skip to avoid duplication.
    if (env.handle != cb_handle) {
        // During replay, env.handle is the handle of the replayer (queue or primary command buffer).
        objlist.add(env.handle);
    }

    objlist.add(cb_handle);
    objlist.add(related_object);
    objlist.add(resource);
    return objlist;
}

bool ReplayCommands(SyncEnvironment& env, AccessContext& access_context, const CommandBufferContext& cb_context,
                    ResourceUsageTag base_tag, const Location& loc) {
    bool skip = false;
    CommandReplayContext replay_context(env, access_context, base_tag);
    const CommandData& command_data = cb_context.GetCommandData();

    for (const CommandEntry& entry : cb_context.GetCommands()) {
        const ResourceUsageTag tag = base_tag + entry.tag;
        std::visit(
            [&](const auto& storage) {
                const auto& command = storage.MakeCommand(command_data);
                using CommandType = std::decay_t<decltype(command)>;
                bool command_skip = false;
                if constexpr (std::is_same_v<CommandType, RenderPassCommand>) {
                    command_skip = command.Validate(replay_context, cb_context, entry.tag, loc);
                    if (!command_skip) {
                        command.Apply(replay_context, tag);
                    }
                } else {
                    AccessContext& current_access_context = replay_context.CurrentAccessContext();
                    command_skip = command.Validate(env, current_access_context, cb_context, entry.tag, loc);
                    if (!command_skip) {
                        command.Apply(env, tag, current_access_context);
                    }
                }
                skip |= command_skip;
            },
            entry.storage);
    }
    return skip;
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

RenderPassCommand::Storage RenderPassCommand::MakeStorage(CommandData& command_data) const {
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

bool RenderPassCommand::Validate(CommandReplayContext& replay_context, const CommandBufferContext& cb_context,
                                 ResourceUsageTag replay_tag, const Location& loc) const {
    switch (type) {
        case Type::kBegin: {
            RenderPassAccessContext* rp_context = replay_context.BeginRenderPass(*this);
            return rp_context ? rp_context->ValidateBeginRenderPass(cb_context, replay_context.GetSyncEnvironment(), command)
                              : false;
        }
        case Type::kNext: {
            const RenderPassAccessContext* rp_context = replay_context.GetRenderPassContext();
            return rp_context ? rp_context->ValidateNextSubpass(cb_context, replay_context.GetSyncEnvironment(), command) : false;
        }
        case Type::kEnd: {
            const RenderPassAccessContext* rp_context = replay_context.GetRenderPassContext();
            return rp_context ? rp_context->ValidateEndRenderPass(cb_context, replay_context.GetSyncEnvironment(), command) : false;
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
            rp_context->RecordBeginRenderPass(tag, tag + 1);
            break;
        case Type::kNext:
            rp_context->RecordNextSubpass(tag, tag + 1, tag + 2, tag + 3);
            break;
        case Type::kEnd:
            rp_context->RecordEndRenderPass(&replay_context.ExternalAccessContext(), tag, tag + 1);
            replay_context.EndRenderPass();
            break;
    }
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
        objlist = BaseObjectList(env, cb_context, buffer.Handle(), query_pool_handle);
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

ImageTransferCommand ImageTransferCommand::Storage::MakeCommand(const CommandData& command_data) const {
    vvl::span<const Access> accesses;
    if (access_count != 0) {
        accesses = vvl::make_span(&command_data.image_transfer_accesses[first_access], access_count);
    }
    return {CommandList<Access>(accesses)};
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
    accesses.AppendTo(command_data.image_transfer_accesses);
    return {first_access, access_count};
}

namespace {

template <typename RegionType>
ImageTransferCommand MakeBufferToImageCopyCommandImpl(const vvl::Buffer* src_buffer, const vvl::Image* dst_image,
                                                      uint32_t region_count, const RegionType* regions) {
    ImageTransferCommand command;
    command.accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_buffer && dst_image) {
            const AccessRange range = MakeRange(region.bufferOffset, dst_image->GetBufferSizeFromCopyImage(region));
            command.accesses.emplace_back(
                ImageTransferCommand::BufferAccess{src_buffer, SYNC_COPY_TRANSFER_READ, range, region_index});
        }
        if (dst_image) {
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{
                dst_image, SYNC_COPY_TRANSFER_WRITE, region.imageSubresource, region.imageOffset, region.imageExtent, region_index});
        }
    }
    return command;
}

template <typename RegionType>
ImageTransferCommand MakeImageToBufferCopyCommandImpl(const vvl::Image* src_image, const vvl::Buffer* dst_buffer,
                                                      uint32_t region_count, const RegionType* regions) {
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
ImageTransferCommand MakeImageBlitCommandImpl(const vvl::Image* src_image, const vvl::Image* dst_image, uint32_t region_count,
                                              const RegionType* regions) {
    ImageTransferCommand command;
    command.accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_image) {
            const VkOffset3D offset = {std::min(region.srcOffsets[0].x, region.srcOffsets[1].x),
                                       std::min(region.srcOffsets[0].y, region.srcOffsets[1].y),
                                       std::min(region.srcOffsets[0].z, region.srcOffsets[1].z)};
            const VkExtent3D extent = {uint32_t(std::abs(region.srcOffsets[1].x - region.srcOffsets[0].x)),
                                       uint32_t(std::abs(region.srcOffsets[1].y - region.srcOffsets[0].y)),
                                       uint32_t(std::abs(region.srcOffsets[1].z - region.srcOffsets[0].z))};
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{
                src_image, SYNC_BLIT_TRANSFER_READ, region.srcSubresource, offset, extent, region_index});
        }
        if (dst_image) {
            const VkOffset3D offset = {std::min(region.dstOffsets[0].x, region.dstOffsets[1].x),
                                       std::min(region.dstOffsets[0].y, region.dstOffsets[1].y),
                                       std::min(region.dstOffsets[0].z, region.dstOffsets[1].z)};
            const VkExtent3D extent = {uint32_t(std::abs(region.dstOffsets[1].x - region.dstOffsets[0].x)),
                                       uint32_t(std::abs(region.dstOffsets[1].y - region.dstOffsets[0].y)),
                                       uint32_t(std::abs(region.dstOffsets[1].z - region.dstOffsets[0].z))};
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{
                dst_image, SYNC_BLIT_TRANSFER_WRITE, region.dstSubresource, offset, extent, region_index});
        }
    }
    return command;
}

template <typename RegionType>
ImageTransferCommand MakeImageResolveCommandImpl(const vvl::Image* src_image, const vvl::Image* dst_image,
                                                 uint32_t region_count, const RegionType* regions) {
    ImageTransferCommand command;
    command.accesses.reserve(2 * region_count);
    for (const auto [region_index, region] : vvl::enumerate(regions, region_count)) {
        if (src_image) {
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{src_image, SYNC_RESOLVE_TRANSFER_READ,
                                                                            region.srcSubresource, region.srcOffset,
                                                                            region.extent, region_index});
        }
        if (dst_image) {
            command.accesses.emplace_back(ImageTransferCommand::ImageAccess{dst_image, SYNC_RESOLVE_TRANSFER_WRITE,
                                                                            region.dstSubresource, region.dstOffset,
                                                                            region.extent, region_index});
        }
    }
    return command;
}

}  // namespace

ImageTransferCommand MakeBufferToImageCopyCommand(const vvl::Buffer* src_buffer, const vvl::Image* dst_image,
                                                  uint32_t region_count, const VkBufferImageCopy* regions) {
    return MakeBufferToImageCopyCommandImpl(src_buffer, dst_image, region_count, regions);
}

ImageTransferCommand MakeBufferToImageCopyCommand(const vvl::Buffer* src_buffer, const vvl::Image* dst_image,
                                                  uint32_t region_count, const VkBufferImageCopy2* regions) {
    return MakeBufferToImageCopyCommandImpl(src_buffer, dst_image, region_count, regions);
}

ImageTransferCommand MakeImageToBufferCopyCommand(const vvl::Image* src_image, const vvl::Buffer* dst_buffer,
                                                  uint32_t region_count, const VkBufferImageCopy* regions) {
    return MakeImageToBufferCopyCommandImpl(src_image, dst_buffer, region_count, regions);
}

ImageTransferCommand MakeImageToBufferCopyCommand(const vvl::Image* src_image, const vvl::Buffer* dst_buffer,
                                                  uint32_t region_count, const VkBufferImageCopy2* regions) {
    return MakeImageToBufferCopyCommandImpl(src_image, dst_buffer, region_count, regions);
}

ImageTransferCommand MakeImageBlitCommand(const vvl::Image* src_image, const vvl::Image* dst_image, uint32_t region_count,
                                          const VkImageBlit* regions) {
    return MakeImageBlitCommandImpl(src_image, dst_image, region_count, regions);
}

ImageTransferCommand MakeImageBlitCommand(const vvl::Image* src_image, const vvl::Image* dst_image, uint32_t region_count,
                                          const VkImageBlit2* regions) {
    return MakeImageBlitCommandImpl(src_image, dst_image, region_count, regions);
}

ImageTransferCommand MakeImageResolveCommand(const vvl::Image* src_image, const vvl::Image* dst_image, uint32_t region_count,
                                             const VkImageResolve* regions) {
    return MakeImageResolveCommandImpl(src_image, dst_image, region_count, regions);
}

ImageTransferCommand MakeImageResolveCommand(const vvl::Image* src_image, const vvl::Image* dst_image, uint32_t region_count,
                                             const VkImageResolve2* regions) {
    return MakeImageResolveCommandImpl(src_image, dst_image, region_count, regions);
}

ImageTransferCommand MakeImageClearCommand(const vvl::Image* image, uint32_t range_count,
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
                    ImageRangeGen range_gen = SubState(*value.image)
                                                  .MakeImageRangeGen(RangeFromLayers(value.subresource), value.offset, value.extent,
                                                                     false);
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

}  // namespace syncval
