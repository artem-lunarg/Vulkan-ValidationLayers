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

#include "sync_barrier.h"
#include "containers/span.h"

struct Location;
struct VulkanTypedHandle;

namespace vvl {
class Buffer;
class Image;
class ImageView;
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
    const T& front() const { return Data().front(); }
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
                 ImageTransferCommand::Storage, BarrierCommand::Storage, EventCommand::Storage, RenderPassCommand::Storage>;

struct CommandData {
    std::vector<std::shared_ptr<const vvl::Buffer>> buffers;
    std::vector<std::shared_ptr<const vvl::Image>> images;
    std::vector<BufferCopyRegion> buffer_copy_regions;
    std::vector<VkImageCopy> image_copy_regions;
    std::vector<ImageTransferCommand::Access> image_transfer_accesses;
    std::vector<BarrierSet> barrier_sets;
    std::vector<std::shared_ptr<const vvl::Event>> events;
    std::vector<BarrierSet> event_barrier_sets;
    std::vector<std::shared_ptr<const vvl::RenderPass>> render_passes;
    std::vector<std::shared_ptr<const vvl::ImageView>> render_pass_attachments;

    uint32_t AddBuffer(const vvl::Buffer& buffer);
    uint32_t AddImage(const vvl::Image& image);
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
