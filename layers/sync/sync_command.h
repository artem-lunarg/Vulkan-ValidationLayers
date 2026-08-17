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

#include "sync/sync_barrier.h"
#include "sync/sync_event.h"
#include "generated/vk_object_types.h"

#include <variant>

struct Location;

namespace vvl {
enum class Func;
class CommandBuffer;
class DescriptorSet;
class Image;
class ImageView;
class Pipeline;
class RenderPass;
}  // namespace vvl

namespace syncval {

class CommandBufferContext;

// These objects own the information needed to validate and apply a command. Record-time
// validation creates a temporary object, while successful recording stores an object in
// the command buffer for future submit-time replay.
struct BufferCopyCommand {
    struct Region {
        AccessRange src_range;
        AccessRange dst_range;
    };

    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    std::shared_ptr<const vvl::Buffer> src_buffer;
    std::shared_ptr<const vvl::Buffer> dst_buffer;
    std::vector<Region> regions;
    uint32_t src_handle_index = vvl::kNoIndex32;
    uint32_t dst_handle_index = vvl::kNoIndex32;
};

BufferCopyCommand MakeBufferCopyCommand(std::shared_ptr<const vvl::Buffer> src_buffer,
                                        std::shared_ptr<const vvl::Buffer> dst_buffer, uint32_t region_count,
                                        const VkBufferCopy* regions);
BufferCopyCommand MakeBufferCopyCommand(std::shared_ptr<const vvl::Buffer> src_buffer,
                                        std::shared_ptr<const vvl::Buffer> dst_buffer, uint32_t region_count,
                                        const VkBufferCopy2* regions);

// Commands that perform a single buffer range access:
// vkCmdFillBuffer, vkCmdUpdateBuffer, vkCmdCopyQueryPoolResults, vkCmdWriteBufferMarkerAMD/2AMD
struct BufferAccessCommand {
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_access);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    std::shared_ptr<const vvl::Buffer> buffer;
    SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
    AccessRange range;
    // SyncFlag::kMarker selects marker hazard detection and the current (render pass aware) context
    uint8_t flags = 0;
    // Additional handle for the error object list (query pool for vkCmdCopyQueryPoolResults)
    VulkanTypedHandle extra_handle;
    uint32_t handle_index = vvl::kNoIndex32;
};

BufferAccessCommand MakeBufferAccessCommand(std::shared_ptr<const vvl::Buffer> buffer, SyncAccessIndex access_index,
                                            const AccessRange& range, uint8_t flags = 0,
                                            const VulkanTypedHandle& extra_handle = VulkanTypedHandle());

// Image transfer commands: vkCmdCopyImage/2, vkCmdCopyBufferToImage/2, vkCmdCopyImageToBuffer/2,
// vkCmdBlitImage/2, vkCmdResolveImage/2, vkCmdClearColorImage, vkCmdClearDepthStencilImage.
// Accesses are resolved from the command parameters when the command is created.
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
        // Single mip range built from the region's subresource layers, or the range
        // provided by the clear commands
        VkImageSubresourceRange subresource_range{};
        // Region geometry. Zero extent means the access covers the whole subresource range
        // (the clear commands). Blit offsets are normalized to offset + extent form.
        VkOffset3D offset{};
        VkExtent3D extent{};
        uint32_t region_index = 0;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    std::vector<ImageAccess> image_accesses;
    std::vector<BufferAccess> buffer_accesses;
    // Validation reports region by region, source accesses before destination accesses,
    // and stops after the first region with a hazard (except the clear commands)
    bool buffer_accesses_first = false;
    bool stop_at_hazardous_region = true;
};

ImageTransferCommand MakeImageCopyCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageCopy* regions);
ImageTransferCommand MakeImageCopyCommand(std::shared_ptr<const vvl::Image> src_image, std::shared_ptr<const vvl::Image> dst_image,
                                          uint32_t region_count, const VkImageCopy2* regions);
// Covers both copy directions: the buffer access direction is defined by buffer_is_source
ImageTransferCommand MakeBufferImageCopyCommand(std::shared_ptr<const vvl::Buffer> buffer, std::shared_ptr<const vvl::Image> image,
                                                bool buffer_is_source, uint32_t region_count, const VkBufferImageCopy* regions);
ImageTransferCommand MakeBufferImageCopyCommand(std::shared_ptr<const vvl::Buffer> buffer, std::shared_ptr<const vvl::Image> image,
                                                bool buffer_is_source, uint32_t region_count, const VkBufferImageCopy2* regions);
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
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_barrier);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    BarrierSet barriers;
};

// One descriptor access resolved by the shader accesses heuristic walk
// (CommandBufferContext::CollectDescriptorAccesses)
struct DescriptorAccess {
    SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
    // Buffer style access: uniform/storage/texel buffers, acceleration structure backing
    std::shared_ptr<const vvl::Buffer> buffer;
    AccessRange range;
    // Image style access
    std::shared_ptr<const vvl::ImageView> image_view;
    // Input attachment reads are restricted to the render area and use raster ordering rules
    bool input_attachment = false;
    AttachmentAccess attachment_access;
    VkOffset3D render_offset{};
    VkExtent3D render_extent{};
    uint32_t handle_index = vvl::kNoIndex32;

    // Reporting information, read only when a hazard is found.
    // TODO: move to a cold side array so replay does not load it (hot/cold split).
    VulkanTypedHandle command_handle;      // registered in the command buffer handle records
    VulkanTypedHandle description_handle;  // named in the error message
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL;
    VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    uint32_t set_number = 0;
    uint32_t binding = 0;
    VkImageLayout image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t set_index = 0;  // into the owning command's descriptor_sets
};

// vkCmdDispatch, vkCmdDispatchBase/KHR, vkCmdDispatchIndirect
struct DispatchCommand {
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    std::vector<DescriptorAccess> descriptor_accesses;

    // vkCmdDispatchIndirect
    std::shared_ptr<const vvl::Buffer> indirect_buffer;
    AccessRange indirect_range;
    uint32_t indirect_handle_index = vvl::kNoIndex32;

    // Reporting information for the descriptor accesses
    std::shared_ptr<const vvl::Pipeline> pipeline;
    std::vector<std::shared_ptr<const vvl::DescriptorSet>> descriptor_sets;
};

DispatchCommand MakeDispatchCommand(const CommandBufferContext& cb_context, VkPipelineBindPoint bind_point,
                                    std::shared_ptr<const vvl::Buffer> indirect_buffer, VkDeviceSize indirect_offset);

// One draw attachment access (color, depth or stencil aspect), resolved at record time.
// The range generator is stored by value: for dynamic rendering the source info is
// destroyed at vkCmdEndRendering. TODO: store the generator inputs instead (memory).
struct DrawAttachmentAccess {
    std::shared_ptr<const vvl::ImageView> view;
    ImageRangeGen range_gen;
    SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
    AttachmentAccess attachment_access;
    // Selects the error message flavor and reporting
    bool dynamic_rendering = false;
    std::string description;  // "color attachment N in subpass S" and similar (legacy render pass)
    // Rebuilds the report location for dynamic rendering, e.g. "pColorAttachments[0].imageView":
    // pColorAttachments (with attachment_index), pDepthAttachment or pStencilAttachment
    vvl::Field attachment_field = vvl::Field::Empty;
    uint32_t attachment_index = 0;
};

// All draw entry points: vkCmdDraw*, including indexed, indirect, count, mesh and multi variants
struct DrawCommand {
    struct VertexAccess {
        std::shared_ptr<const vvl::Buffer> buffer;
        AccessRange range;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    std::vector<DescriptorAccess> descriptor_accesses;
    std::vector<VertexAccess> vertex_accesses;
    std::vector<VertexAccess> index_accesses;
    std::vector<DrawAttachmentAccess> attachment_accesses;

    std::shared_ptr<const vvl::Buffer> indirect_buffer;
    std::vector<AccessRange> indirect_ranges;
    uint32_t indirect_handle_index = vvl::kNoIndex32;

    std::shared_ptr<const vvl::Buffer> count_buffer;
    AccessRange count_range;
    uint32_t count_handle_index = vvl::kNoIndex32;
    const char* count_label = "count";

    // Reporting information for the descriptor accesses
    std::shared_ptr<const vvl::Pipeline> pipeline;
    std::vector<std::shared_ptr<const vvl::DescriptorSet>> descriptor_sets;
};

// Descriptor and attachment accesses of the currently bound graphics state
DrawCommand MakeDrawCommand(const CommandBufferContext& cb_context);
// The fixed function pieces differ between the draw entry points and are added separately
void AddDrawIndirectAccess(DrawCommand& command, std::shared_ptr<const vvl::Buffer> buffer, VkDeviceSize offset,
                           VkDeviceSize struct_size, uint32_t draw_count, uint32_t stride);
void AddDrawCountAccess(DrawCommand& command, std::shared_ptr<const vvl::Buffer> buffer, VkDeviceSize offset,
                        const char* count_label);

// Shader binding table region resolved to its backing buffer
struct SbtAccess {
    std::shared_ptr<const vvl::Buffer> buffer;
    AccessRange range;
    const char* label = nullptr;  // raygen/miss/hit/callable
    uint32_t handle_index = vvl::kNoIndex32;
};

// vkCmdTraceRays*: descriptor accesses, shader binding table reads and the indirect parameters read
struct TraceRaysCommand {
    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    std::vector<DescriptorAccess> descriptor_accesses;
    std::vector<SbtAccess> sbt_accesses;

    std::shared_ptr<const vvl::Buffer> indirect_buffer;
    AccessRange indirect_range;
    uint32_t indirect_handle_index = vvl::kNoIndex32;

    // Reporting information for the descriptor accesses
    std::shared_ptr<const vvl::Pipeline> pipeline;
    std::vector<std::shared_ptr<const vvl::DescriptorSet>> descriptor_sets;
};

// Descriptor accesses of the currently bound ray tracing state. The shader binding
// table and indirect accesses need device address resolution and are added by the caller.
TraceRaysCommand MakeTraceRaysCommand(const CommandBufferContext& cb_context);

// vkCmdBuildAccelerationStructuresKHR and vkCmdCopyAccelerationStructure* commands:
// a flattened list of buffer accesses resolved at record time (scratch buffer,
// geometry inputs and the buffers that back the acceleration structures).
struct AccelerationStructureCommand {
    struct BufferAccess {
        std::shared_ptr<const vvl::Buffer> buffer;
        AccessRange range;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        // Reporting: an acceleration structure buffer access references the structure and
        // its parameter field path; other accesses are reported as plain buffer errors.
        VkAccelerationStructureKHR acceleration_structure = VK_NULL_HANDLE;
        std::string description;
        // The source of a self-referencing build (src == dst) is validated but not
        // applied: the destination write replaces the access anyway.
        bool apply = true;
        uint32_t handle_index = vvl::kNoIndex32;
    };

    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    std::vector<BufferAccess> accesses;
};

// vkCmdDecodeVideoKHR / vkCmdEncodeVideoKHR: the bitstream buffer access and the
// picture accesses (output/input picture, reconstructed picture, reference pictures
// and the encode quantization map), resolved to range generators at record time.
struct VideoCommand {
    struct PictureAccess {
        std::shared_ptr<const vvl::ImageView> view;  // reporting handle
        std::shared_ptr<const vvl::Image> image;     // keeps the range generator's image alive
        ImageRangeGen range_gen;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        std::string description;
    };

    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    std::shared_ptr<const vvl::Buffer> bitstream_buffer;
    AccessRange bitstream_range;
    SyncAccessIndex bitstream_access_index = SYNC_ACCESS_INDEX_NONE;
    std::string bitstream_description;
    uint32_t bitstream_handle_index = vvl::kNoIndex32;

    std::vector<PictureAccess> picture_accesses;
};

// vkCmdClearAttachments: attachment clears resolved against the active render pass
// or dynamic rendering state at record time; one access per cleared aspect group,
// clear rect and (with multiview) view index.
struct ClearAttachmentsCommand {
    struct ClearAccess {
        std::shared_ptr<const vvl::ImageView> view;
        ImageRangeGen range_gen;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        AttachmentAccess attachment_access;
        // Reporting information
        VkImageAspectFlags aspect_mask = 0;
        uint32_t rect_index = 0;
        VkClearRect rect = {};
        std::string description;
    };

    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    std::vector<ClearAccess> accesses;
};

// vkCmdBeginRendering (load operations) and vkCmdEndRendering (resolve and store
// operations): attachment accesses resolved to range generators at record time.
struct DynamicRenderingCommand {
    enum class OpType { kLoad, kResolveRead, kResolveWrite, kStore };
    struct Access {
        std::shared_ptr<const vvl::ImageView> view;
        ImageRangeGen range_gen;
        SyncAccessIndex access_index = SYNC_ACCESS_INDEX_NONE;
        AttachmentAccess attachment_access;
        // Reporting information; op stores the VkAttachmentLoadOp / VkResolveModeFlagBits /
        // VkAttachmentStoreOp value that corresponds to op_type.
        OpType op_type = OpType::kLoad;
        uint32_t op = 0;
        std::string description;
    };

    bool Validate(const CommandBufferContext& cb_context, const Location& loc) const;
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, const CommandBufferContext& cb_context,
                  vvl::Func command, const ResourceUsageRange& record_time_validated_tags, const Location& loc) const;
    void Record(CommandBufferContext& cb_context, ResourceUsageTag tag, bool apply_accesses);
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    std::vector<Access> accesses;
};

// vkCmdBeginRenderPass/2: owns the state to reconstruct a render pass access context
// during submit replay. Covers two tags: the subpass 0 layout transitions and the load
// operations. NextSubpassCommand/EndRenderPassCommand operate on the reconstructed
// context; they are dispatched explicitly by the replay loop because they switch the
// access context that the commands inside the render pass are replayed against.
struct BeginRenderPassCommand {
    std::shared_ptr<const vvl::RenderPass> rp_state;
    VkRect2D render_area = {};
    std::vector<std::shared_ptr<const vvl::ImageView>> attachment_views;
    uint32_t render_pass_instance_id = 0;
};

// vkCmdNextSubpass/2. Four tags: resolve and store operations of the previous subpass,
// then the layout transitions and load operations of the next subpass.
struct NextSubpassCommand {};

// vkCmdEndRenderPass/2. Two tags: store/resolve operations and the final layout transitions.
struct EndRenderPassCommand {};

// One executed secondary command buffer of a vkCmdExecuteCommands call. The secondary's
// recorded command stream is replayed recursively. Covers the index tag plus the range of
// the secondary's access log that record time imports into the primary.
struct ExecuteCommandsCommand {
    std::shared_ptr<const vvl::CommandBuffer> secondary_cb;
};

// vkCmdSetEvent/2, vkCmdResetEvent/2, vkCmdWaitEvents/2.
// Unlike the access commands, record time application has different semantics than submit
// replay (the set event first scope snapshot), so it stays in the event record helpers and
// this command participates only in submit replay.
using EventOperation = std::variant<SetEventReplay, ResetEventReplay, WaitEventsReplay>;

struct EventCommand {
    bool Validate(const SyncEnvironment& env, const AccessContext& access_context, ResourceUsageTag tag,
                  const ResourceUsageRange& record_time_validated_tags) const;
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag tag) const;

    EventOperation operation;
};

using RecordedCommand = std::variant<BufferCopyCommand, BufferAccessCommand, ImageTransferCommand, DispatchCommand, DrawCommand,
                                     TraceRaysCommand, AccelerationStructureCommand, VideoCommand, ClearAttachmentsCommand,
                                     DynamicRenderingCommand, BeginRenderPassCommand, NextSubpassCommand, EndRenderPassCommand,
                                     ExecuteCommandsCommand, PipelineBarrierCommand, EventCommand>;

struct RecordedCommandEntry {
    // The template avoids a GCC issue with forwarding directly into std::variant.
    template <typename Command>
    RecordedCommandEntry(ResourceUsageTag tag, uint32_t tag_count, Command&& command)
        : tag(tag), tag_count(tag_count), command(std::forward<Command>(command)) {}

    ResourceUsageTag tag;
    uint32_t tag_count;  // number of consecutive tags covered by this command (render pass commands cover several)
    RecordedCommand command;
};

bool ReplayRecordedCommands(SyncEnvironment& env, AccessContext& access_context, const CommandBufferContext& cb_context,
                            ResourceUsageTag base_tag, const ResourceUsageRange& record_time_validated_tags, const Location& loc);

}  // namespace syncval
