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
#pragma once

#include "sync/sync_access_state.h"
#include "sync/sync_barrier.h"
#include "containers/span.h"
#include "vulkan/generated/vk_object_types.h"

#include <cassert>
#include <iterator>
#include <optional>
#include <string>
#include <variant>
#include <vector>

struct Location;

namespace vvl {
enum class Func;
class Buffer;
class DescriptorSet;
class AccelerationStructureKHR;
class Event;
class Image;
class ImageView;
class Pipeline;
class RenderPass;
}  // namespace vvl

namespace syncval {

class CommandBufferContext;
class CommandReplayContext;

struct ResourceAccessCommand;
struct ImageTransferCommand;
struct PipelineBarrierCommand;
struct EventCommand;
struct RenderPassCommand;
struct CommandData;

struct BufferCopyRegion {
    VkDeviceSize src_offset;
    VkDeviceSize dst_offset;
    VkDeviceSize size;
};

// Owns elements while a command is assembled and becomes a view when the command is reconstructed from CommandData.
// The command interface consequently stays the same for immediate validation and submit-time replay, while persistent
// variable-sized data is stored in shared side arrays rather than in one allocation per command.
template <typename T>
class CommandList {
  public:
    CommandList() = default;
    explicit CommandList(vvl::span<const T> view) : view_(view) {}
    CommandList& operator=(const std::vector<T>& values) {
        assert(!view_.data());
        owned_ = values;
        return *this;
    }

    const T* begin() const { return Data().begin(); }
    const T* end() const { return Data().end(); }
    size_t size() const { return Data().size(); }
    bool empty() const { return Data().empty(); }
    operator vvl::span<const T>() const { return Data(); }

    void reserve(size_t capacity) {
        assert(!view_.data());
        owned_.reserve(capacity);
    }

    template <typename... Args>
    T& emplace_back(Args&&... args) {
        assert(!view_.data());
        return owned_.emplace_back(std::forward<Args>(args)...);
    }

    T& front() {
        assert(!view_.data());
        return owned_.front();
    }
    const T& front() const { return Data().front(); }
    T& back() {
        assert(!view_.data());
        return owned_.back();
    }
    const T& back() const { return Data().back(); }

    std::vector<T>& Mutable() {
        assert(!view_.data());
        return owned_;
    }

    void Append(CommandList&& other) {
        assert(!view_.data() && !other.view_.data());
        owned_.reserve(owned_.size() + other.owned_.size());
        std::move(other.owned_.begin(), other.owned_.end(), std::back_inserter(owned_));
    }

    void AppendTo(std::vector<T>& destination) const& {
        const auto data = Data();
        destination.insert(destination.end(), data.begin(), data.end());
    }

    void AppendTo(std::vector<T>& destination) && {
        if (view_.data()) {
            destination.insert(destination.end(), view_.begin(), view_.end());
        } else {
            destination.insert(destination.end(), std::make_move_iterator(owned_.begin()), std::make_move_iterator(owned_.end()));
        }
    }

  private:
    vvl::span<const T> Data() const { return view_.data() ? view_ : vvl::make_span(owned_); }

    std::vector<T> owned_;
    vvl::span<const T> view_;
};

struct BufferCopyCommand {
    const vvl::Buffer& src_buffer;
    const vvl::Buffer& dst_buffer;
    vvl::span<const BufferCopyRegion> regions;

    struct Storage {
        uint32_t src_buffer_index;
        uint32_t dst_buffer_index;
        uint32_t first_region;
        uint32_t region_count;
        uint32_t src_handle_index;
        uint32_t dst_handle_index;

        BufferCopyCommand MakeCommand(const CommandData& command_data) const;
    };

    Storage MakeStorage(CommandData& command_data, uint32_t src_handle_index, uint32_t dst_handle_index) const;
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                  ResourceUsageTag replay_tag = kInvalidTag) const;
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTagEx src_tag_ex,
               ResourceUsageTagEx dst_tag_ex) const;
};

struct BufferAccessCommand {
    const vvl::Buffer& buffer;
    SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
    AccessRange range;
    uint8_t flags = 0;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    const char* resource_name = "buffer ";

    struct Storage {
        uint32_t buffer_index;
        SyncAccessIndex access_index;
        AccessRange range;
        uint8_t flags;
        VkQueryPool query_pool;
        const char* resource_name;
        uint32_t handle_index;

        BufferAccessCommand MakeCommand(const CommandData& command_data) const;
    };

    Storage MakeStorage(CommandData& command_data, uint32_t handle_index) const;
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                  ResourceUsageTag replay_tag = kInvalidTag) const;
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTagEx tag_ex) const;
};

BufferAccessCommand MakeBufferAccessCommand(const vvl::Buffer& buffer, SyncAccessIndex access_index, AccessRange range,
                                            uint8_t flags = 0, VkQueryPool query_pool = VK_NULL_HANDLE,
                                            const char* resource_name = "buffer ");

// A lossless, ordered list of accesses resolved while recording a Vulkan command.
// Unlike AccessContext, this object is an event stream rather than a synchronization
// state summary. It can therefore be applied for record-time validation, submit-time
// validation, or both without reconstructing accesses from compacted state.
struct ResourceAccessCommand {
    enum class DescriptorResourceType { kBuffer, kImage, kAccelerationStructure };

    struct DescriptorInfo {
        std::shared_ptr<const vvl::Pipeline> pipeline;
        std::shared_ptr<const vvl::DescriptorSet> descriptor_set;
        VulkanTypedHandle resource_handle;
        DescriptorResourceType resource_type = DescriptorResourceType::kBuffer;
        uint32_t set = 0;
        VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
        uint32_t binding = 0;
        uint32_t array_element = 0;
        VkShaderStageFlagBits stage = VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
        VkImageLayout image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct BufferAccess {
        struct AccelerationStructureInfo {
            std::shared_ptr<const vvl::AccelerationStructureKHR> acceleration_structure;
            std::string location;
        };

        std::shared_ptr<const vvl::Buffer> buffer;
        std::shared_ptr<const vvl::Pipeline> pipeline;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        AccessRange range;
        SyncFlags flags = 0;
        bool apply_access = true;
        // Preserve legacy record-time object lists that intentionally reported only the accessed resource.
        bool legacy_record_time_object_only = false;
        VulkanTypedHandle tag_handle;
        std::string resource_name = "buffer ";
        std::optional<DescriptorInfo> descriptor_info;
        std::optional<AccelerationStructureInfo> acceleration_structure_info;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    struct ImageViewAccess {
        enum class ErrorLocation { kNone, kColorAttachment, kDepthAttachment, kStencilAttachment };

        std::shared_ptr<const vvl::ImageView> image_view;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        bool use_render_area = false;
        VkOffset3D offset{};
        VkExtent3D extent{};
        uint32_t view_mask = 0;
        VkImageAspectFlags aspect_mask = 0;
        AttachmentAccess attachment_access;
        VulkanTypedHandle tag_handle;
        // Some legacy messages include a related object in addition to the accessed image view (for example, its image).
        VulkanTypedHandle additional_object;
        std::optional<DescriptorInfo> descriptor_info;
        std::string resource_description;
        const char* message_type = "ImageAccessError";
        ErrorLocation error_location = ErrorLocation::kNone;
        uint32_t attachment_index = vvl::kNoIndex32;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    struct ImageRangeAccess {
        enum class ErrorType { kGeneric, kClearAttachment, kVideo };

        std::shared_ptr<const vvl::Image> image;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        VkImageSubresourceRange subresource_range{};
        bool is_depth_sliced = false;
        bool use_offset_extent = false;
        VkOffset3D offset{};
        VkExtent3D extent{};
        AttachmentAccess attachment_access;
        VulkanTypedHandle tag_handle;
        std::string resource_description;
        ErrorType error_type = ErrorType::kGeneric;
        VkImageAspectFlags clear_aspects = 0;
        uint32_t clear_rect_index = 0;
        VkClearRect clear_rect{};
        uint32_t handle_index = vvl::kNoIndex32;
    };

    using Access = std::variant<BufferAccess, ImageViewAccess, ImageRangeAccess>;

    struct Storage {
        uint32_t first_access;
        uint32_t access_count;
        ResourceAccessCommand MakeCommand(const CommandData& command_data) const;
    };

    Storage MakeStorage(CommandData& command_data) const&;
    Storage MakeStorage(CommandData& command_data) &&;

    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                  ResourceUsageTag replay_tag = kInvalidTag) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    void Append(ResourceAccessCommand&& other);

    CommandList<Access> accesses;
};

struct ImageTransferCommand {
    struct BufferAccess {
        std::shared_ptr<const vvl::Buffer> buffer;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        AccessRange range;
        uint32_t region_index = 0;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    struct ImageAccess {
        std::shared_ptr<const vvl::Image> image;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        VkImageSubresourceLayers subresource{};
        VkOffset3D offset{};
        VkExtent3D extent{};
        uint32_t region_index = 0;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    struct ImageRangeAccess {
        std::shared_ptr<const vvl::Image> image;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        VkImageSubresourceRange subresource_range{};
        uint32_t range_index = 0;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    using Access = std::variant<BufferAccess, ImageAccess, ImageRangeAccess>;

    struct Storage {
        uint32_t first_access;
        uint32_t access_count;
        ImageTransferCommand MakeCommand(const CommandData& command_data) const;
    };

    Storage MakeStorage(CommandData& command_data) const&;
    Storage MakeStorage(CommandData& command_data) &&;

    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                  ResourceUsageTag replay_tag = kInvalidTag) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    CommandList<Access> accesses;
};

ImageTransferCommand MakeImageCopyCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageCopy* regions);
ImageTransferCommand MakeImageCopyCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageCopy2* regions);
ImageTransferCommand MakeBufferToImageCopyCommand(std::shared_ptr<const vvl::Buffer> src_buffer,
                                                  std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                                  const VkBufferImageCopy* regions);
ImageTransferCommand MakeBufferToImageCopyCommand(std::shared_ptr<const vvl::Buffer> src_buffer,
                                                  std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                                  const VkBufferImageCopy2* regions);
ImageTransferCommand MakeImageToBufferCopyCommand(std::shared_ptr<const vvl::Image> src_image,
                                                  std::shared_ptr<const vvl::Buffer> dst_buffer, uint32_t region_count,
                                                  const VkBufferImageCopy* regions);
ImageTransferCommand MakeImageToBufferCopyCommand(std::shared_ptr<const vvl::Image> src_image,
                                                  std::shared_ptr<const vvl::Buffer> dst_buffer, uint32_t region_count,
                                                  const VkBufferImageCopy2* regions);
ImageTransferCommand MakeImageBlitCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageBlit* regions);
ImageTransferCommand MakeImageBlitCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageBlit2* regions);
ImageTransferCommand MakeImageResolveCommand(std::shared_ptr<const vvl::Image> src_image,
                                             std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                             const VkImageResolve* regions);
ImageTransferCommand MakeImageResolveCommand(std::shared_ptr<const vvl::Image> src_image,
                                             std::shared_ptr<const vvl::Image> dst_image, uint32_t region_count,
                                             const VkImageResolve2* regions);
ImageTransferCommand MakeImageClearCommand(std::shared_ptr<const vvl::Image> image, uint32_t range_count,
                                           const VkImageSubresourceRange* ranges);

struct PipelineBarrierCommand {
    struct Storage {
        uint32_t command_index;
        const PipelineBarrierCommand& MakeCommand(const CommandData& command_data) const;
    };

    Storage MakeStorage(CommandData& command_data) const&;
    Storage MakeStorage(CommandData& command_data) &&;
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                  ResourceUsageTag replay_tag = kInvalidTag) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_barrier);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    BarrierSet barriers;
};

struct EventCommand {
    enum class Type { kSet, kReset, kWait };

    struct Storage {
        Type type;
        uint32_t first_event;
        uint32_t event_count;
        SyncExecScope exec_scope;
        uint32_t first_barrier_set;
        uint32_t barrier_set_count;
        vvl::Func command;

        EventCommand MakeCommand(const CommandData& command_data) const;
    };

    Storage MakeStorage(CommandData& command_data) const&;
    Storage MakeStorage(CommandData& command_data) &&;

    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, ResourceUsageTag tag,
                  const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_command);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    Type type = Type::kSet;
    CommandList<std::shared_ptr<const vvl::Event>> events;
    SyncExecScope exec_scope;
    CommandList<BarrierSet> barrier_sets;
    vvl::Func command{};
};

struct NoOpCommand {
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc,
                  ResourceUsageTag replay_tag = kInvalidTag) const;
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;
};

struct RenderPassCommand {
    enum class Type { kBegin, kNext, kEnd };

    struct Storage {
        Type type;
        uint32_t render_pass_index;
        uint32_t first_attachment;
        uint32_t attachment_count;
        VkRect2D render_area;
        uint32_t render_pass_instance_id;
        vvl::Func command;

        RenderPassCommand MakeCommand(const CommandData& command_data) const;
    };

    Storage MakeStorage(CommandData& command_data) const&;
    Storage MakeStorage(CommandData& command_data) &&;

    ResourceUsageTag Record(CommandBufferContext& cb_context, bool apply_command) const;
    bool Validate(CommandReplayContext& replay_context, const CommandBufferContext& cb_context,
                  const ResourceUsageRange& record_time_validated_tags, ResourceUsageTag replay_tag,
                  const Location& loc) const;
    void Apply(CommandReplayContext& replay_context, ResourceUsageTag tag) const;

    Type type = Type::kBegin;
    std::shared_ptr<const vvl::RenderPass> render_pass;
    CommandList<std::shared_ptr<const vvl::ImageView>> attachments;
    VkRect2D render_area{};
    uint32_t render_pass_instance_id = 0;
    vvl::Func command{};
};

struct CommandData {
    std::vector<std::shared_ptr<const vvl::Buffer>> buffers;
    std::vector<BufferCopyRegion> buffer_copy_regions;
    std::vector<ResourceAccessCommand::Access> resource_accesses;
    std::vector<ImageTransferCommand::Access> image_transfer_accesses;
    std::vector<PipelineBarrierCommand> pipeline_barrier_commands;
    std::vector<std::shared_ptr<const vvl::Event>> events;
    std::vector<BarrierSet> event_barrier_sets;
    std::vector<std::shared_ptr<const vvl::RenderPass>> render_passes;
    std::vector<std::shared_ptr<const vvl::ImageView>> render_pass_attachments;

    uint32_t AddBuffer(const vvl::Buffer& buffer);
};

using RecordedCommand =
    std::variant<BufferCopyCommand::Storage, BufferAccessCommand::Storage, ResourceAccessCommand::Storage,
                 ImageTransferCommand::Storage, PipelineBarrierCommand::Storage, EventCommand::Storage, NoOpCommand,
                 RenderPassCommand::Storage>;

struct RecordedCommandEntry {
    // The template avoids a GCC issue with forwarding directly into std::variant.
    template <typename Command>
    RecordedCommandEntry(ResourceUsageTag tag, Command&& command) : tag(tag), command(std::forward<Command>(command)) {}

    ResourceUsageTag tag;
    RecordedCommand command;
};

bool ReplayRecordedCommands(SyncEnvironment& env, AccessContext& access_context, const CommandBufferContext& cb_context,
                            ResourceUsageTag base_tag, const ResourceUsageRange& record_time_validated_tags, const Location& loc);

// Reports a hazard found while replaying recorded commands: at queue submission, or when
// vkCmdExecuteCommands validates an executed secondary command buffer against the primary.
// Reproduces the legacy first-use report: message shape, object list and resource description.
bool ReportReplayHazard(const SyncEnvironment& env, const CommandBufferContext& cb_context, const HazardResult& hazard,
                        ResourceUsageTag replay_tag, const VulkanTypedHandle& resource_handle, const Location& loc);
void ApplyRecordedCommands(SyncEnvironment& env, AccessContext& access_context, const CommandBufferContext& cb_context,
                           ResourceUsageTag base_tag);

}  // namespace syncval
