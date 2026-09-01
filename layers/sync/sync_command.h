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

#pragma once

#include "sync_access_state.h"
#include "sync_barrier.h"
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
class Buffer;
class AccelerationStructureKHR;
class DescriptorSet;
class Image;
class ImageView;
class Pipeline;
class RenderPass;
class Event;
enum class Func;
}  // namespace vvl

namespace syncval {

class AccessContext;
class CommandBufferContext;
class CommandReplayContext;
struct CommandData;
struct SyncEnvironment;

struct BufferCopyRegion {
    VkDeviceSize src_offset;
    VkDeviceSize dst_offset;
    VkDeviceSize size;
};

// Owns elements while a command is assembled and becomes a view when reconstructed from CommandData.
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

    std::vector<T>& Mutable() {
        assert(!view_.data());
        return owned_;
    }

    void AppendTo(std::vector<T>& destination) const {
        const auto data = Data();
        destination.insert(destination.end(), data.begin(), data.end());
    }

    void Append(CommandList&& other) {
        assert(!view_.data());
        const auto values = other.Data();
        owned_.insert(owned_.end(), std::make_move_iterator(values.begin()), std::make_move_iterator(values.end()));
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
    uint32_t src_handle_index = vvl::kNoIndex32;
    uint32_t dst_handle_index = vvl::kNoIndex32;

    struct Storage {
        uint32_t src_buffer_index;
        uint32_t dst_buffer_index;
        uint32_t first_region;
        uint32_t region_count;
        uint32_t src_handle_index;
        uint32_t dst_handle_index;
        BufferCopyCommand MakeCommand(const CommandData& command_data) const;
    };
    Storage MakeStorage(CommandData& command_data) const;
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  ResourceUsageTag replay_tag, const Location& loc) const;
    void Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const;
};

struct BufferAccessCommand {
    const vvl::Buffer& buffer;
    SyncAccessIndex access_index;
    AccessRange range;
    uint8_t flags = 0;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    const char* resource_name = "buffer ";
    uint32_t handle_index = vvl::kNoIndex32;

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
    Storage MakeStorage(CommandData& command_data) const;
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  ResourceUsageTag replay_tag, const Location& loc) const;
    void Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const;
};

struct ImageCopyCommand {
    const vvl::Image& src_image;
    const vvl::Image& dst_image;
    vvl::span<const VkImageCopy> regions;
    uint32_t src_handle_index = vvl::kNoIndex32;
    uint32_t dst_handle_index = vvl::kNoIndex32;

    struct Storage {
        uint32_t src_image_index;
        uint32_t dst_image_index;
        uint32_t first_region;
        uint32_t region_count;
        uint32_t src_handle_index;
        uint32_t dst_handle_index;
        ImageCopyCommand MakeCommand(const CommandData& command_data) const;
    };
    Storage MakeStorage(CommandData& command_data) const;
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  ResourceUsageTag replay_tag, const Location& loc) const;
    void Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const;
};

// An ordered list of resource accesses resolved while recording a Vulkan command. Unlike AccessContext, this is an event
// stream rather than a synchronization-state summary, so it can be applied during recording or replay.
struct ResourceAccessCommand {
    enum class DescriptorResourceType { kBuffer, kImage, kAccelerationStructure };

    struct DescriptorInfo {
        const vvl::Pipeline* pipeline = nullptr;
        const vvl::DescriptorSet* descriptor_set = nullptr;
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
            const vvl::AccelerationStructureKHR* acceleration_structure = nullptr;
            std::string location;
        };

        const vvl::Buffer* buffer = nullptr;
        const vvl::Pipeline* pipeline = nullptr;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        AccessRange range;
        SyncFlags flags = 0;
        bool apply_access = true;
        bool legacy_record_time_object_only = false;
        VulkanTypedHandle tag_handle;
        std::string resource_name = "buffer ";
        std::optional<DescriptorInfo> descriptor_info;
        std::optional<AccelerationStructureInfo> acceleration_structure_info;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    struct ImageViewAccess {
        enum class ErrorLocation { kNone, kColorAttachment, kDepthAttachment, kStencilAttachment };

        const vvl::ImageView* image_view = nullptr;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        bool use_render_area = false;
        VkOffset3D offset{};
        VkExtent3D extent{};
        uint32_t view_mask = 0;
        VkImageAspectFlags aspect_mask = 0;
        AttachmentAccess attachment_access;
        VulkanTypedHandle tag_handle;
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

        const vvl::Image* image = nullptr;
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
    CommandList<Access> accesses;

    struct Storage {
        uint32_t first_access;
        uint32_t access_count;
        ResourceAccessCommand MakeCommand(const CommandData& command_data) const;
    };
    Storage MakeStorage(CommandData& command_data) const;
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  ResourceUsageTag replay_tag, const Location& loc) const;
    void Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const;
    void Append(ResourceAccessCommand&& other);
};

struct ImageTransferCommand {
    struct BufferAccess {
        const vvl::Buffer* buffer;
        SyncAccessIndex access_index;
        AccessRange range;
        uint32_t region_index;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    struct ImageAccess {
        const vvl::Image* image;
        SyncAccessIndex access_index;
        VkImageSubresourceLayers subresource;
        VkOffset3D offset;
        VkExtent3D extent;
        uint32_t region_index;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    struct ImageRangeAccess {
        const vvl::Image* image;
        SyncAccessIndex access_index;
        VkImageSubresourceRange subresource_range;
        uint32_t range_index;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    using Access = std::variant<BufferAccess, ImageAccess, ImageRangeAccess>;

    CommandList<Access> accesses;

    struct Storage {
        uint32_t first_access;
        uint32_t access_count;
        ImageTransferCommand MakeCommand(const CommandData& command_data) const;
    };
    Storage MakeStorage(CommandData& command_data) const;
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  ResourceUsageTag replay_tag, const Location& loc) const;
    void Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const;
};

ImageTransferCommand MakeBufferToImageCopyCommand(const vvl::Buffer* src_buffer, const vvl::Image* dst_image,
                                                  uint32_t region_count, const VkBufferImageCopy* regions);
ImageTransferCommand MakeBufferToImageCopyCommand(const vvl::Buffer* src_buffer, const vvl::Image* dst_image,
                                                  uint32_t region_count, const VkBufferImageCopy2* regions);
ImageTransferCommand MakeImageToBufferCopyCommand(const vvl::Image* src_image, const vvl::Buffer* dst_buffer,
                                                  uint32_t region_count, const VkBufferImageCopy* regions);
ImageTransferCommand MakeImageToBufferCopyCommand(const vvl::Image* src_image, const vvl::Buffer* dst_buffer,
                                                  uint32_t region_count, const VkBufferImageCopy2* regions);
ImageTransferCommand MakeImageBlitCommand(const vvl::Image* src_image, const vvl::Image* dst_image, uint32_t region_count,
                                          const VkImageBlit* regions);
ImageTransferCommand MakeImageBlitCommand(const vvl::Image* src_image, const vvl::Image* dst_image, uint32_t region_count,
                                          const VkImageBlit2* regions);
ImageTransferCommand MakeImageResolveCommand(const vvl::Image* src_image, const vvl::Image* dst_image, uint32_t region_count,
                                             const VkImageResolve* regions);
ImageTransferCommand MakeImageResolveCommand(const vvl::Image* src_image, const vvl::Image* dst_image, uint32_t region_count,
                                             const VkImageResolve2* regions);
ImageTransferCommand MakeImageClearCommand(const vvl::Image* image, uint32_t range_count,
                                           const VkImageSubresourceRange* ranges);

struct BarrierCommand {
    const BarrierSet& barrier_set;

    struct Storage {
        uint32_t barrier_set_index;
        BarrierCommand MakeCommand(const CommandData& command_data) const;
    };
    Storage MakeStorage(CommandData& command_data) const;
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  ResourceUsageTag replay_tag, const Location& loc) const;
    void Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const;
};

struct EventCommand {
    enum class Type { kSet, kReset, kWait };

    Type type = Type::kSet;
    CommandList<std::shared_ptr<const vvl::Event>> events;
    SyncExecScope exec_scope;
    CommandList<BarrierSet> barrier_sets;
    vvl::Func command{};

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
    Storage MakeStorage(CommandData& command_data) const;
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  ResourceUsageTag replay_tag, const Location& loc) const;
    void Apply(SyncEnvironment& env, ResourceUsageTag tag, AccessContext& access_context) const;
};

struct RenderPassCommand {
    enum class Type { kBegin, kNext, kEnd };

    Type type;
    std::shared_ptr<const vvl::RenderPass> render_pass;
    CommandList<std::shared_ptr<const vvl::ImageView>> attachments;
    VkRect2D render_area{};
    uint32_t render_pass_instance_id = 0;
    vvl::Func command{};

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
    Storage MakeStorage(CommandData& command_data) const;
    bool Validate(CommandReplayContext& replay_context, const CommandBufferContext& cb_context,
                  ResourceUsageTag replay_tag, const Location& loc) const;
    void Apply(CommandReplayContext& replay_context, ResourceUsageTag tag) const;
};

using CommandStorage =
    std::variant<BufferCopyCommand::Storage, BufferAccessCommand::Storage, ImageCopyCommand::Storage,
                 ResourceAccessCommand::Storage, ImageTransferCommand::Storage, BarrierCommand::Storage, EventCommand::Storage,
                 RenderPassCommand::Storage>;

struct CommandData {
    std::vector<std::shared_ptr<const vvl::Buffer>> buffers;
    std::vector<std::shared_ptr<const vvl::Image>> images;
    std::vector<BufferCopyRegion> buffer_copy_regions;
    std::vector<VkImageCopy> image_copy_regions;
    std::vector<ResourceAccessCommand::Access> resource_accesses;
    std::vector<ImageTransferCommand::Access> image_transfer_accesses;
    std::vector<BarrierSet> barrier_sets;
    std::vector<std::shared_ptr<const vvl::Event>> events;
    std::vector<BarrierSet> event_barrier_sets;
    std::vector<std::shared_ptr<const vvl::RenderPass>> render_passes;
    std::vector<std::shared_ptr<const vvl::ImageView>> render_pass_attachments;
    std::vector<std::shared_ptr<const vvl::ImageView>> image_views;
    std::vector<std::shared_ptr<const vvl::Pipeline>> pipelines;
    std::vector<std::shared_ptr<const vvl::DescriptorSet>> descriptor_sets;
    std::vector<std::shared_ptr<const vvl::AccelerationStructureKHR>> acceleration_structures;

    uint32_t AddBuffer(const vvl::Buffer& buffer);
    uint32_t AddImage(const vvl::Image& image);
    void AddImageView(const vvl::ImageView& image_view);
    void AddPipeline(const vvl::Pipeline& pipeline);
    void AddDescriptorSet(const vvl::DescriptorSet& descriptor_set);
    void AddAccelerationStructure(const vvl::AccelerationStructureKHR& acceleration_structure);
};

// TODO: CommandEntry won't be needed after all commands are introduced.
// Tag could be derived from command index. Remove entry type when and
// use array of commands instead.
struct CommandEntry {
    ResourceUsageTag tag;
    uint32_t tag_count;
    CommandStorage storage;
};

bool ReplayCommands(SyncEnvironment& env, AccessContext& access_context, const CommandBufferContext& cb_context,
                    ResourceUsageTag base_tag, const Location& loc);

}  // namespace syncval
