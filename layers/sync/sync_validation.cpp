/* Copyright (c) 2019-2026 The Khronos Group Inc.
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

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "sync/sync_error_messages.h"
#include "sync/sync_validation.h"
#include "sync/sync_event.h"
#include "sync/sync_image.h"
#include "state_tracker/buffer_state.h"
#include "state_tracker/ray_tracing_state.h"
#include "state_tracker/render_pass_state.h"
#include "utils/convert_utils.h"
#include "utils/image_utils.h"
#include "utils/ray_tracing_utils.h"
#include "utils/text_utils.h"
#include "vk_layer_config.h"
#include "containers/tls_guard.h"

namespace syncval {

static bool GetShowStatsEnvVar() {
    // Set environment variable as non zero number to enable stats reporting
    const auto show_stats_str = GetEnvironment("VK_SYNCVAL_SHOW_STATS");
    return !show_stats_str.empty() && std::atoi(show_stats_str.c_str()) != 0;
}

SyncValidator::SyncValidator(vvl::DispatchDevice* dev, syncval::Instance* instance_vo)
    : BaseClass(dev, instance_vo, LayerObjectTypeSyncValidation), error_messages_(*this), report_stats_(GetShowStatsEnvVar()) {}

SyncValidator::~SyncValidator() {
    // Instance level SyncValidator does not have much to say
    const bool device_validation_object = (device != nullptr);

    if (device_validation_object && report_stats_) {
        stats.ReportOnDestruction();
    }
}

// Location to add per-queue submit debug info if built with -D DEBUG_CAPTURE_KEYBOARD=ON.
void SyncValidator::DebugCapture() {
    if (report_stats_) {
        stats.UpdateAccessStats(*this);

        // NOTE: mimalloc stats are not updated here - mostly because they are tracked
        // per thread and updating stats only for current thread feels a bit unbalanced.
        // Instead we have specific places to trigger memory stats collection.

        const std::string report = stats.CreateReport();
        std::cout << report;
#ifdef VK_USE_PLATFORM_WIN32_KHR
        OutputDebugString(report.c_str());
#endif
    }
}

bool SyncValidator::SyncError(SyncHazard hazard, const LogObjectList& objlist, const Location& loc,
                              const std::string& error_message) const {
    return LogError(string_SyncHazardVUID(hazard), objlist, loc, "%s", error_message.c_str());
}

ResourceUsageRange SyncValidator::ReserveGlobalTagRange(size_t tag_count) const {
    ResourceUsageRange reserve;
    reserve.begin = tag_limit_.fetch_add(tag_count);
    reserve.end = reserve.begin + tag_count;
    return reserve;
}

void SyncValidator::EnsureTimelineSignalsLimit(uint32_t signals_per_queue_limit, QueueId queue) {
    for (auto& [_, signals] : timeline_signals_) {
        const size_t initial_signal_count = signals.size();
        vvl::unordered_map<QueueId, uint32_t> signals_per_queue;
        for (const SignalInfo& signal : signals) {
            ++signals_per_queue[signal.first_scope.queue];
        }
        const bool filter_queue = queue != kQueueIdInvalid;
        for (auto it = signals.begin(); it != signals.end();) {
            if (filter_queue && it->first_scope.queue != queue) {
                ++it;
                continue;
            }
            auto& counter = signals_per_queue[it->first_scope.queue];
            if (counter > signals_per_queue_limit) {
                it = signals.erase(it);
                --counter;
            } else {
                ++it;
            }
        }
        stats.RemoveTimelineSignals(uint32_t(initial_signal_count - signals.size()));
    }
}

void SyncValidator::ApplySignalsUpdate(SignalsUpdate& update, const BatchContextPtr& last_batch) {
    // NOTE: All conserved QueueBatchContexts need to have their access logs reset to use the global
    // logger and the only conserved QBCs are those referenced by unwaited signals and the last batch.

    for (auto& signal_entry : update.binary_signal_requests) {
        auto& signal_batch = signal_entry.second.batch;
        // Batches retained for signalled semaphore don't need to retain
        // event data, unless it's the last batch in the submit
        if (signal_batch != last_batch) {
            signal_batch->ResetEventsContext();
            // Make sure that retained batches are minimal, and trim
            // after the events contexts has been cleared.
            signal_batch->Trim();
        }
        const VkSemaphore semaphore = signal_entry.first;
        SignalInfo& signal_info = signal_entry.second;
        binary_signals_.insert_or_assign(semaphore, std::move(signal_info));
    }
    for (VkSemaphore semaphore : update.binary_unsignal_requests) {
        binary_signals_.erase(semaphore);
    }
    for (auto& [semaphore, new_signals] : update.timeline_signals) {
        std::vector<SignalInfo>& signals = timeline_signals_[semaphore];
        vvl::Append(signals, new_signals);
        stats.AddTimelineSignals((uint32_t)new_signals.size());

        // Update host sync points
        std::deque<TimelineHostSyncPoint>& host_sync_points = host_waitable_semaphores_[semaphore];
        for (SignalInfo& new_signal : new_signals) {
            if (new_signal.batch) {
                // The lifetimes of the semaphore host sync points are managed by vkWaitSemaphores.
                // kMaxTimelineHostSyncPoints limit is used when the program does not use vkWaitSemaphores.
                // We accumulate up to kMaxTimelineHostSyncPoints of the host sync points per semaphore.
                // Dropping old sync points cannot introduce false positives but may miss a sync hazard.
                // The limit is chosen to be large enough comparing to typical numbers of queue submissions
                // between host synchronization points.
                const uint32_t kMaxTimelineHostSyncPoints = 256;  // max ~6 Kb per semaphore
                if (host_sync_points.size() >= kMaxTimelineHostSyncPoints) {
                    host_sync_points.pop_front();
                }
                // Add a host sync point for this signal
                TimelineHostSyncPoint sync_point;
                assert(new_signal.first_scope.queue != kQueueIdInvalid);
                sync_point.queue_id = new_signal.first_scope.queue;
                sync_point.tag = new_signal.batch->GetTagRange().end - 1;
                sync_point.timeline_value = new_signal.timeline_value;
                host_sync_points.emplace_back(sync_point);
            }
        }
    }
    for (const auto& remove_signals_request : update.remove_timeline_signals_requests) {
        auto& signals = timeline_signals_[remove_signals_request.semaphore];
        for (auto it = signals.begin(); it != signals.end();) {
            const SignalInfo& signal = *it;
            if (signal.first_scope.queue == remove_signals_request.queue &&
                signal.timeline_value < remove_signals_request.signal_threshold_value) {
                it = signals.erase(it);
                stats.RemoveTimelineSignals(1);
                continue;
            }
            ++it;
        }
    }

    // Enforce max signals limit in case timeline is signaled multiple times and never/rarely is waited on.
    // This does not introduce errors/false-positives (check EnsureTimelineSignalsLimit documentation)
    const uint32_t kMaxTimelineSignalsPerQueue = 100;
    EnsureTimelineSignalsLimit(kMaxTimelineSignalsPerQueue);
}

void SyncValidator::ApplyTaggedWait(QueueId queue_id, ResourceUsageTag tag,
                                    const LastSynchronizedPresent& last_synchronized_present,
                                    const std::vector<ResourceUsageTag>& queue_sync_tags) {
    assert(queue_id < queue_id_limit_);
    assert(queue_sync_tags.empty() || queue_sync_tags.size() == queue_id_limit_);
    assert(queue_sync_tags.empty() || queue_sync_tags[queue_id] == tag);

    // Create a list of queues that have to be synchronized up to some point.
    // Note that, in general, not only the queue_id queue has to be synchronized.
    // If queue_id was synchronized with other queues through a semaphore wait,
    // then waiting for queue_id also means waiting for those other queues
    std::vector<std::pair<QueueId, ResourceUsageTag>> sync_points;
    if (queue_sync_tags.empty()) {
        sync_points.emplace_back(queue_id, tag);
    } else {
        sync_points.reserve(queue_id_limit_);
        for (const auto [sync_queue, sync_tag] : vvl::enumerate(queue_sync_tags)) {
            if (sync_tag > 0) {
                sync_points.emplace_back((QueueId)sync_queue, sync_tag);
            }
        }
    }

    const auto all_batches = GetAllQueueBatchContexts();
    for (const auto& batch : all_batches) {
        for (const auto& [sync_queue, sync_tag] : sync_points) {
            batch->ApplyTaggedWait(sync_queue, sync_tag, last_synchronized_present);
        }
        batch->Trim();
    }
}

void SyncValidator::ApplyAcquireWait(const AcquiredImage& acquired) {
    for (const auto& batch : GetAllQueueBatchContexts()) {
        batch->ApplyAcquireWait(acquired);
        batch->Trim();
    }
}

std::vector<BatchContextPtr> SyncValidator::GetAllQueueBatchContexts() {
    // Get last batch from each queue
    std::vector<BatchContextPtr> batch_contexts = GetLastBatches([](auto) { return true; });

    // Get batches from binary signals
    for (auto& [_, signal] : binary_signals_) {
        if (!vvl::Contains(batch_contexts, signal.batch)) {
            batch_contexts.emplace_back(signal.batch);
        }
    }
    // Get batches from timeline signals
    for (auto& [_, signals] : timeline_signals_) {
        for (const auto& signal : signals) {
            if (signal.batch && !vvl::Contains(batch_contexts, signal.batch)) {
                batch_contexts.emplace_back(signal.batch);
            }
        }
    }
    // Get present batches
    device_state->ForEachShared<vvl::Swapchain>([&batch_contexts](const std::shared_ptr<vvl::Swapchain>& swapchain) {
        auto& sync_swapchain = SubState(*swapchain);
        sync_swapchain.GetPresentBatches(batch_contexts);
    });

    return batch_contexts;
}

void SyncValidator::UpdateFenceHostSyncPoint(VkFence fence, FenceHostSyncPoint&& sync_point) {
    std::shared_ptr<const vvl::Fence> fence_state = Get<vvl::Fence>(fence);
    if (!vvl::StateObject::Invalid(fence_state)) {
        waitable_fences_[fence_state->VkHandle()] = std::move(sync_point);
    }
}

void SyncValidator::WaitForFence(VkFence fence) {
    if (auto fence_it = waitable_fences_.find(fence); fence_it != waitable_fences_.end()) {
        FenceHostSyncPoint& wait_for = fence_it->second;
        if (wait_for.queue_id != kQueueIdInvalid) {
            const QueueState& queue_state = GetQueueState(wait_for.queue_id);
            ApplyTaggedWait(wait_for.queue_id, wait_for.tag, queue_state.GetLastSynchronizedPresent(), wait_for.queue_sync_tags);
        } else if (!vvl::StateObject::Invalid(wait_for.acquired.image)) {
            ApplyAcquireWait(wait_for.acquired);
        }
        waitable_fences_.erase(fence_it);
    }
}

void SyncValidator::WaitForSemaphore(VkSemaphore semaphore, uint64_t value) {
    std::deque<TimelineHostSyncPoint>* sync_points = vvl::Find(host_waitable_semaphores_, semaphore);
    if (!sync_points) {
        return;
    }
    auto matching_sync_point = [value](const TimelineHostSyncPoint& sync_point) { return sync_point.timeline_value >= value; };
    auto sync_point_it = std::find_if(sync_points->begin(), sync_points->end(), matching_sync_point);
    if (sync_point_it == sync_points->end()) {
        return;
    }

    const TimelineHostSyncPoint& sync_point = *sync_point_it;
    const QueueState& queue_state = GetQueueState(sync_point.queue_id);

    // TODO: specify queue sync tags argument similar to WaitForFence
    ApplyTaggedWait(sync_point.queue_id, sync_point.tag, queue_state.GetLastSynchronizedPresent(), {});

    // Remove signals before the resolving one (keep the resolving signal).
    std::vector<SignalInfo>& signals = timeline_signals_[semaphore];
    const size_t initial_signal_count = signals.size();
    vvl::erase_if(signals, [&sync_point](SignalInfo& signal) {
        return signal.first_scope.queue == sync_point.queue_id && signal.timeline_value < sync_point.timeline_value;
    });
    stats.RemoveTimelineSignals(uint32_t(initial_signal_count - signals.size()));

    // We can remove all sync points that are in the scope of current wait.
    // Subsequent attempts to synchronize on the host with already synchronized
    // timeline values will result in noop.
    sync_points->erase(sync_points->begin(), sync_point_it + 1 /* include resolving sync point too*/);
}

void SyncValidator::UpdateSyncImageMemoryBindState(uint32_t count, const VkBindImageMemoryInfo* infos) {
    for (const auto& info : vvl::make_span(infos, count)) {
        if (VK_NULL_HANDLE == info.image) continue;
        auto image_state = Get<vvl::Image>(info.image);

        // Need to protect if some VkBindMemoryStatus are not VK_SUCCESS
        if (!image_state->HasBeenBound()) continue;

        auto& sub_state = SubState(*image_state);
        if (sub_state.IsTiled()) {
            sub_state.SetOpaqueBaseAddress(*device_state);
        }
    }
}

QueueId SyncValidator::GetQueueId(VkQueue queue) const {
    for (const QueueState& queue_state : queue_states_) {
        if (queue_state.GetQueue()->VkHandle() == queue) {
            return queue_state.GetQueueId();
        }
    }
    return kQueueIdInvalid;
}

QueueState& SyncValidator::GetQueueState(QueueId queue_id) {
    assert(queue_id != kQueueIdInvalid);
    assert(queue_id < queue_id_limit_);
    return queue_states_[queue_id];
}

void SyncValidator::Created(vvl::CommandBuffer& cb_state) {
    cb_state.SetSubState(container_type, std::make_unique<CommandBufferSubState>(*this, cb_state));
}

void SyncValidator::Created(vvl::Swapchain& swapchain_state) {
    swapchain_state.SetSubState(container_type, std::make_unique<SwapchainSubState>(swapchain_state));
}

void SyncValidator::Created(vvl::Image& image_state) {
    image_state.SetSubState(container_type, std::make_unique<ImageSubState>(image_state));
}

void SyncValidator::PreCallRecordDestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* pAllocator,
                                               const RecordObject& record_obj) {
    if (const auto buffer_state = Get<vvl::Buffer>(buffer)) {
        const VkDeviceSize base_address = ResourceBaseAddress(*buffer_state);
        const AccessRange buffer_range(base_address, base_address + buffer_state->GetSize());
        for (const auto& batch : GetAllQueueBatchContexts()) {
            batch->OnResourceDestroyed(buffer_range);
            batch->Trim();
        }
    }
}

void SyncValidator::PreCallRecordDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator,
                                              const RecordObject& record_obj) {
    if (const auto image_state = Get<vvl::Image>(image)) {
        for (const auto& batch : GetAllQueueBatchContexts()) {
            const auto& sub_state = SubState(*image_state);
            ImageRangeGen range_gen = sub_state.MakeImageRangeGen(image_state->full_range, false);
            for (; range_gen->non_empty(); ++range_gen) {
                const AccessRange subresource_range = *range_gen;
                batch->OnResourceDestroyed(subresource_range);
            }
            batch->Trim();
        }
    }
}

void SyncValidator::PreCallRecordDestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain,
                                                     const VkAllocationCallbacks* pAllocator, const RecordObject& record_obj) {
    for (const auto& batch : GetAllQueueBatchContexts()) {
        batch->last_synchronized_present.OnDestroySwapchain(swapchain);
    }
}

bool SyncValidator::PreCallValidateCmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
                                                 uint32_t regionCount, const VkBufferCopy* pRegions,
                                                 const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    auto src_buffer = Get<vvl::Buffer>(srcBuffer);
    auto dst_buffer = Get<vvl::Buffer>(dstBuffer);
    if (!src_buffer || !dst_buffer) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    small_vector<BufferCopyRegion, 1> regions;
    regions.reserve(regionCount);
    for (const VkBufferCopy& region : vvl::make_span(pRegions, regionCount)) {
        regions.emplace_back(BufferCopyRegion{region.srcOffset, region.dstOffset, region.size});
    }
    const BufferCopyCommand command{*src_buffer, *dst_buffer, regions};
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdCopyBuffer2(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2* pCopyBufferInfo,
                                                  const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    if (!pCopyBufferInfo) {
        return false;
    }
    auto src_buffer = Get<vvl::Buffer>(pCopyBufferInfo->srcBuffer);
    auto dst_buffer = Get<vvl::Buffer>(pCopyBufferInfo->dstBuffer);
    if (!src_buffer || !dst_buffer) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    small_vector<BufferCopyRegion, 1> regions;
    regions.reserve(pCopyBufferInfo->regionCount);
    for (const VkBufferCopy2& region : vvl::make_span(pCopyBufferInfo->pRegions, pCopyBufferInfo->regionCount)) {
        regions.emplace_back(BufferCopyRegion{region.srcOffset, region.dstOffset, region.size});
    }
    const BufferCopyCommand command{*src_buffer, *dst_buffer, regions};
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdCopyBuffer2KHR(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2KHR* pCopyBufferInfo,
                                                     const ErrorObject& error_obj) const {
    return PreCallValidateCmdCopyBuffer2(commandBuffer, pCopyBufferInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdCopyImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
                                                VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount,
                                                const VkImageCopy* pRegions, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    auto src_image = Get<vvl::Image>(srcImage);
    auto dst_image = Get<vvl::Image>(dstImage);
    if (!src_image || !dst_image) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    const ImageCopyCommand command{*src_image, *dst_image, vvl::make_span(pRegions, regionCount)};
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdCopyImage2(VkCommandBuffer commandBuffer, const VkCopyImageInfo2* pCopyImageInfo,
                                                 const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    if (!pCopyImageInfo) {
        return false;
    }
    auto src_image = Get<vvl::Image>(pCopyImageInfo->srcImage);
    auto dst_image = Get<vvl::Image>(pCopyImageInfo->dstImage);
    if (!src_image || !dst_image) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    small_vector<VkImageCopy, 1> regions;
    regions.reserve(pCopyImageInfo->regionCount);
    for (const VkImageCopy2& region : vvl::make_span(pCopyImageInfo->pRegions, pCopyImageInfo->regionCount)) {
        regions.emplace_back(
            VkImageCopy{region.srcSubresource, region.srcOffset, region.dstSubresource, region.dstOffset, region.extent});
    }
    const ImageCopyCommand command{*src_image, *dst_image, regions};
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdCopyImage2KHR(VkCommandBuffer commandBuffer, const VkCopyImageInfo2KHR* pCopyImageInfo,
                                                    const ErrorObject& error_obj) const {
    return PreCallValidateCmdCopyImage2(commandBuffer, pCopyImageInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdPipelineBarrier(
    VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
    VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier* pImageMemoryBarriers, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    const SyncExecScope src_exec_scope(SyncExecScope::MakeSrc(queue_flags, srcStageMask));
    const SyncExecScope dst_exec_scope(SyncExecScope::MakeDst(queue_flags, dstStageMask));
    const BarrierSet barrier_set(*this, src_exec_scope, dst_exec_scope, memoryBarrierCount, pMemoryBarriers,
                                 bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
    stats.OnBarrierCommand(memoryBarrierCount, bufferMemoryBarrierCount, imageMemoryBarrierCount,
                           barrier_set.execution_dependency_barrier_count);

    const BarrierCommand command{barrier_set};
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::RecordCmdPipelineBarrier(CommandBufferContext& cb_context, BarrierSet&& barrier_set,
                                             const Location& loc) const {
    const ResourceUsageTag tag = cb_context.NextCommandTag(loc.function);
    for (const SyncBufferBarrier& buffer_barrier : barrier_set.buffer_barriers) {
        cb_context.AddCommandHandle(tag, buffer_barrier.buffer->Handle());
    }
    for (SyncImageBarrier& image_barrier : barrier_set.image_barriers) {
        if (image_barrier.layout_transition) {
            const ResourceUsageTagEx tag_ex = cb_context.AddCommandHandle(tag, image_barrier.image->Handle());
            image_barrier.handle_index = tag_ex.handle_index;
        }
    }
    const BarrierCommand command{barrier_set};
    if (syncval_settings.IsRecordTimeValidationEnabled()) {
        AccessContext& access_context = cb_context.GetCurrentAccessContext();
        command.Apply(cb_context.GetSyncEnvironment(), tag, access_context);
    }
    if (syncval_settings.full_validation) {
        cb_context.StoreCommand(tag, command);
    }
    // TODO: Keep this unconditional because secondary command buffer validation and
    // recording still use replay entries. Revisit when secondaries use the command model.
    cb_context.AddReplayEntry(tag, true, PipelineBarrierReplay(std::move(barrier_set)));
}

void SyncValidator::PostCallRecordCmdPipelineBarrier(
    VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
    VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier* pImageMemoryBarriers, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    const SyncExecScope src_exec_scope(SyncExecScope::MakeSrc(queue_flags, srcStageMask));
    const SyncExecScope dst_exec_scope(SyncExecScope::MakeDst(queue_flags, dstStageMask));

    BarrierSet barrier_set(*this, src_exec_scope, dst_exec_scope, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount,
                           pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
    RecordCmdPipelineBarrier(cb_context, std::move(barrier_set), record_obj.location);
}

bool SyncValidator::PreCallValidateCmdPipelineBarrier2KHR(VkCommandBuffer commandBuffer, const VkDependencyInfoKHR* pDependencyInfo,
                                                          const ErrorObject& error_obj) const {
    return PreCallValidateCmdPipelineBarrier2(commandBuffer, pDependencyInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdPipelineBarrier2(VkCommandBuffer commandBuffer, const VkDependencyInfo* pDependencyInfo,
                                                       const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    if (!pDependencyInfo) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    const BarrierSet barrier_set(*this, queue_flags, *pDependencyInfo);
    stats.OnBarrierCommand(pDependencyInfo->memoryBarrierCount, pDependencyInfo->bufferMemoryBarrierCount,
                           pDependencyInfo->imageMemoryBarrierCount, barrier_set.execution_dependency_barrier_count);

    const BarrierCommand command{barrier_set};
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdPipelineBarrier2KHR(VkCommandBuffer commandBuffer, const VkDependencyInfoKHR* pDependencyInfo,
                                                         const RecordObject& record_obj) {
    PostCallRecordCmdPipelineBarrier2(commandBuffer, pDependencyInfo, record_obj);
}

void SyncValidator::PostCallRecordCmdPipelineBarrier2(VkCommandBuffer commandBuffer, const VkDependencyInfo* pDependencyInfo,
                                                      const RecordObject& record_obj) {
    if (!pDependencyInfo) {
        return;
    }
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();
    BarrierSet barrier_set(*this, queue_flags, *pDependencyInfo);
    RecordCmdPipelineBarrier(cb_context, std::move(barrier_set), record_obj.location);
}

void SyncValidator::FinishDeviceSetup(const VkDeviceCreateInfo* pCreateInfo, const Location& loc) {
    // The state tracker sets up the device state
    BaseClass::FinishDeviceSetup(pCreateInfo, loc);

    // Returns queues in the same order as advertised by the driver.
    // This allows to have deterministic QueueId between runs that simplifies debugging.
    auto get_sorted_queues = [this]() {
        std::vector<std::shared_ptr<vvl::Queue>> queues;
        device_state->ForEachShared<vvl::Queue>(
            [&queues](const std::shared_ptr<vvl::Queue>& queue) { queues.emplace_back(queue); });
        std::sort(queues.begin(), queues.end(), [](const auto& q1, const auto& q2) {
            return (q1->queue_family_index < q2->queue_family_index) ||
                   (q1->queue_family_index == q2->queue_family_index && q1->queue_index < q2->queue_index);
        });
        return queues;
    };
    queue_states_.reserve(device_state->Count<vvl::Queue>());
    for (const auto& queue : get_sorted_queues()) {
        queue_states_.emplace_back(QueueState(queue, queue_id_limit_++));
    }

    const auto env_debug_command_number = GetEnvironment("VK_SYNCVAL_DEBUG_COMMAND_NUMBER");
    if (!env_debug_command_number.empty()) {
        debug_command_number = static_cast<uint32_t>(std::stoul(env_debug_command_number));
    }
    const auto env_debug_reset_count = GetEnvironment("VK_SYNCVAL_DEBUG_RESET_COUNT");
    if (!env_debug_reset_count.empty()) {
        debug_reset_count = static_cast<uint32_t>(std::stoul(env_debug_reset_count));
    }
    debug_cmdbuf_pattern = GetEnvironment("VK_SYNCVAL_DEBUG_CMDBUF_PATTERN");
    text::ToLower(debug_cmdbuf_pattern);
}

void SyncValidator::PreCallRecordDestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator,
                                               const RecordObject& record_obj) {
    queue_states_.clear();
    binary_signals_.clear();
    timeline_signals_.clear();
    waitable_fences_.clear();
    host_waitable_semaphores_.clear();
}

void SyncValidator::PostCallRecordCreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo* pCreateInfo,
                                                  const VkAllocationCallbacks* pAllocator, VkSemaphore* pSemaphore,
                                                  const RecordObject& record_obj) {
    if (record_obj.result != VK_SUCCESS) {
        return;
    }
    assert(!vvl::Contains(timeline_signals_, *pSemaphore));
}

void SyncValidator::PreCallRecordDestroySemaphore(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks* pAllocator,
                                                  const RecordObject& record_obj) {
    if (auto sem_state = Get<vvl::Semaphore>(semaphore); sem_state && (sem_state->type == VK_SEMAPHORE_TYPE_TIMELINE)) {
        if (auto it = timeline_signals_.find(semaphore); it != timeline_signals_.end()) {
            stats.RemoveTimelineSignals((uint32_t)it->second.size());
            timeline_signals_.erase(it);
        }
    }
}

bool SyncValidator::ValidateBeginRenderPass(VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo* pRenderPassBegin,
                                            const VkSubpassBeginInfo* pSubpassBeginInfo, const ErrorObject& error_obj) const {
    bool skip = false;
    if (!pRenderPassBegin || !syncval_settings.IsRecordTimeValidationEnabled()) {
        return skip;
    }
    auto rp_state = Get<vvl::RenderPass>(pRenderPassBegin->renderPass);
    if (!rp_state) {
        return skip;
    }
    auto fb_state = Get<vvl::Framebuffer>(pRenderPassBegin->framebuffer);
    if (!fb_state) {
        return skip;
    }

    const std::vector<std::shared_ptr<const vvl::ImageView>> attachments =
        device_state->GetAttachmentViews(*pRenderPassBegin, *fb_state);
    if (attachments.empty()) {
        return skip;
    }

    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    RenderPassCommand command{RenderPassCommand::Type::kBegin, std::move(rp_state)};
    command.attachments = attachments;
    command.render_area = pRenderPassBegin->renderArea;
    command.render_pass_instance_id = cb_context.GetCurrentRenderPassInstanceId();
    command.command = error_obj.location.function;
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdBeginRenderPass(VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo* pRenderPassBegin,
                                                      VkSubpassContents contents, const ErrorObject& error_obj) const {
    VkSubpassBeginInfo subpass_begin_info = vku::InitStructHelper();
    subpass_begin_info.contents = contents;
    return ValidateBeginRenderPass(commandBuffer, pRenderPassBegin, &subpass_begin_info, error_obj);
}

bool SyncValidator::PreCallValidateCmdBeginRenderPass2(VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo* pRenderPassBegin,
                                                       const VkSubpassBeginInfo* pSubpassBeginInfo,
                                                       const ErrorObject& error_obj) const {
    return ValidateBeginRenderPass(commandBuffer, pRenderPassBegin, pSubpassBeginInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdBeginRenderPass2KHR(VkCommandBuffer commandBuffer,
                                                          const VkRenderPassBeginInfo* pRenderPassBegin,
                                                          const VkSubpassBeginInfo* pSubpassBeginInfo,
                                                          const ErrorObject& error_obj) const {
    return PreCallValidateCmdBeginRenderPass2(commandBuffer, pRenderPassBegin, pSubpassBeginInfo, error_obj);
}

bool SyncValidator::ValidateCmdNextSubpass(VkCommandBuffer commandBuffer, const VkSubpassBeginInfo* pSubpassBeginInfo,
                                           const VkSubpassEndInfo* pSubpassEndInfo, const ErrorObject& error_obj) const {
    bool skip = false;
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return skip;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const RenderPassCommand command{RenderPassCommand::Type::kNext, {}, {}, {}, 0, error_obj.location.function};
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdNextSubpass(VkCommandBuffer commandBuffer, VkSubpassContents contents,
                                                  const ErrorObject& error_obj) const {
    // Convert to a NextSubpass2
    VkSubpassBeginInfo subpass_begin_info = vku::InitStructHelper();
    subpass_begin_info.contents = contents;
    VkSubpassEndInfo subpass_end_info = vku::InitStructHelper();
    return ValidateCmdNextSubpass(commandBuffer, &subpass_begin_info, &subpass_end_info, error_obj);
}

bool SyncValidator::PreCallValidateCmdNextSubpass2KHR(VkCommandBuffer commandBuffer, const VkSubpassBeginInfo* pSubpassBeginInfo,
                                                      const VkSubpassEndInfo* pSubpassEndInfo, const ErrorObject& error_obj) const {
    return PreCallValidateCmdNextSubpass2(commandBuffer, pSubpassBeginInfo, pSubpassEndInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdNextSubpass2(VkCommandBuffer commandBuffer, const VkSubpassBeginInfo* pSubpassBeginInfo,
                                                   const VkSubpassEndInfo* pSubpassEndInfo, const ErrorObject& error_obj) const {
    return ValidateCmdNextSubpass(commandBuffer, pSubpassBeginInfo, pSubpassEndInfo, error_obj);
}

bool SyncValidator::ValidateCmdEndRenderPass(VkCommandBuffer commandBuffer, const ErrorObject& error_obj) const {
    bool skip = false;
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return skip;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const RenderPassCommand command{RenderPassCommand::Type::kEnd, {}, {}, {}, 0, error_obj.location.function};
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdEndRenderPass(VkCommandBuffer commandBuffer, const ErrorObject& error_obj) const {
    return ValidateCmdEndRenderPass(commandBuffer, error_obj);
}

bool SyncValidator::PreCallValidateCmdEndRenderPass2(VkCommandBuffer commandBuffer, const VkSubpassEndInfo* pSubpassEndInfo,
                                                     const ErrorObject& error_obj) const {
    return ValidateCmdEndRenderPass(commandBuffer, error_obj);
}

bool SyncValidator::PreCallValidateCmdEndRenderPass2KHR(VkCommandBuffer commandBuffer, const VkSubpassEndInfo* pSubpassEndInfo,
                                                        const ErrorObject& error_obj) const {
    return PreCallValidateCmdEndRenderPass2(commandBuffer, pSubpassEndInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdBeginRenderingKHR(VkCommandBuffer commandBuffer, const VkRenderingInfoKHR* pRenderingInfo,
                                                        const ErrorObject& error_obj) const {
    return PreCallValidateCmdBeginRendering(commandBuffer, pRenderingInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdBeginRendering(VkCommandBuffer commandBuffer, const VkRenderingInfo* pRenderingInfo,
                                                     const ErrorObject& error_obj) const {
    bool skip = false;
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    if (!pRenderingInfo) return skip;

    vvl::TlsGuard<BeginRenderingCmdState> cmd_state(&skip, std::move(cb_state));
    cmd_state->AddRenderingInfo(*this, *pRenderingInfo);

    // We need to set skip, because the TlsGuard destructor is looking at the skip value for RAII cleanup.
    skip |= GetCommandBufferContext(*cmd_state->cb_state).ValidateBeginRendering(error_obj, *cmd_state);
    return skip;
}

void SyncValidator::PostCallRecordCmdBeginRenderingKHR(VkCommandBuffer commandBuffer, const VkRenderingInfoKHR* pRenderingInfo,
                                                       const RecordObject& record_obj) {
    PostCallRecordCmdBeginRendering(commandBuffer, pRenderingInfo, record_obj);
}

void SyncValidator::PostCallRecordCmdBeginRendering(VkCommandBuffer commandBuffer, const VkRenderingInfo* pRenderingInfo,
                                                    const RecordObject& record_obj) {
    vvl::TlsGuard<BeginRenderingCmdState> cmd_state;

    assert(cmd_state && cmd_state->cb_state && (cmd_state->cb_state->VkHandle() == commandBuffer));
    // Note: for fine grain locking need to to something other than cast.
    auto cb_state = std::const_pointer_cast<vvl::CommandBuffer>(cmd_state->cb_state);
    GetCommandBufferContext(*cb_state).RecordBeginRendering(*cmd_state, record_obj.location);
}

bool SyncValidator::PreCallValidateCmdEndRenderingKHR(VkCommandBuffer commandBuffer, const ErrorObject& error_obj) const {
    return PreCallValidateCmdEndRendering(commandBuffer, error_obj);
}

bool SyncValidator::PreCallValidateCmdEndRendering(VkCommandBuffer commandBuffer, const ErrorObject& error_obj) const {
    bool skip = false;
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    skip |= GetCommandBufferContext(*cb_state).ValidateEndRendering(error_obj);
    return skip;
}

void SyncValidator::PreCallRecordCmdEndRenderingKHR(VkCommandBuffer commandBuffer, const RecordObject& record_obj) {
    PreCallRecordCmdEndRendering(commandBuffer, record_obj);
}

void SyncValidator::PreCallRecordCmdEndRendering(VkCommandBuffer commandBuffer, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    GetCommandBufferContext(*cb_state).RecordEndRendering(record_obj);
}

template <typename RegionType>
bool SyncValidator::ValidateCmdCopyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage,
                                                 VkImageLayout dstImageLayout, uint32_t regionCount, const RegionType* pRegions,
                                                 const Location& loc) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto src_buffer = Get<vvl::Buffer>(srcBuffer);
    auto dst_image = Get<vvl::Image>(dstImage);
    const ImageTransferCommand command =
        MakeBufferToImageCopyCommand(src_buffer.get(), dst_image.get(), regionCount, pRegions);
    return command.Validate(cb_context, loc);
}

bool SyncValidator::PreCallValidateCmdCopyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage,
                                                        VkImageLayout dstImageLayout, uint32_t regionCount,
                                                        const VkBufferImageCopy* pRegions, const ErrorObject& error_obj) const {
    return ValidateCmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions,
                                        error_obj.location);
}

bool SyncValidator::PreCallValidateCmdCopyBufferToImage2KHR(VkCommandBuffer commandBuffer,
                                                            const VkCopyBufferToImageInfo2KHR* pCopyBufferToImageInfo,
                                                            const ErrorObject& error_obj) const {
    return PreCallValidateCmdCopyBufferToImage2(commandBuffer, pCopyBufferToImageInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdCopyBufferToImage2(VkCommandBuffer commandBuffer,
                                                         const VkCopyBufferToImageInfo2* pCopyBufferToImageInfo,
                                                         const ErrorObject& error_obj) const {
    return ValidateCmdCopyBufferToImage(commandBuffer, pCopyBufferToImageInfo->srcBuffer, pCopyBufferToImageInfo->dstImage,
                                        pCopyBufferToImageInfo->dstImageLayout, pCopyBufferToImageInfo->regionCount,
                                        pCopyBufferToImageInfo->pRegions, error_obj.location.dot(Field::pCopyBufferToImageInfo));
}

template <typename RegionType>
bool SyncValidator::ValidateCmdCopyImageToBuffer(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
                                                 VkBuffer dstBuffer, uint32_t regionCount, const RegionType* pRegions,
                                                 const Location& loc) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto src_image = Get<vvl::Image>(srcImage);
    auto dst_buffer = Get<vvl::Buffer>(dstBuffer);
    const VkDeviceMemory dst_memory = (dst_buffer && !dst_buffer->sparse) ? dst_buffer->MemoryState()->VkHandle() : VK_NULL_HANDLE;
    const vvl::Buffer* dst_buffer_access = dst_memory != VK_NULL_HANDLE ? dst_buffer.get() : nullptr;
    const ImageTransferCommand command =
        MakeImageToBufferCopyCommand(src_image.get(), dst_buffer_access, regionCount, pRegions);
    return command.Validate(cb_context, loc);
}

bool SyncValidator::PreCallValidateCmdCopyImageToBuffer(VkCommandBuffer commandBuffer, VkImage srcImage,
                                                        VkImageLayout srcImageLayout, VkBuffer dstBuffer, uint32_t regionCount,
                                                        const VkBufferImageCopy* pRegions, const ErrorObject& error_obj) const {
    return ValidateCmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions,
                                        error_obj.location);
}

bool SyncValidator::PreCallValidateCmdCopyImageToBuffer2KHR(VkCommandBuffer commandBuffer,
                                                            const VkCopyImageToBufferInfo2KHR* pCopyImageToBufferInfo,
                                                            const ErrorObject& error_obj) const {
    return PreCallValidateCmdCopyImageToBuffer2(commandBuffer, pCopyImageToBufferInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdCopyImageToBuffer2(VkCommandBuffer commandBuffer,
                                                         const VkCopyImageToBufferInfo2* pCopyImageToBufferInfo,
                                                         const ErrorObject& error_obj) const {
    return ValidateCmdCopyImageToBuffer(commandBuffer, pCopyImageToBufferInfo->srcImage, pCopyImageToBufferInfo->srcImageLayout,
                                        pCopyImageToBufferInfo->dstBuffer, pCopyImageToBufferInfo->regionCount,
                                        pCopyImageToBufferInfo->pRegions, error_obj.location.dot(Field::pCopyImageToBufferInfo));
}

template <typename RegionType>
bool SyncValidator::ValidateCmdBlitImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
                                         VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount,
                                         const RegionType* pRegions, VkFilter filter, const Location& loc) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto src_image = Get<vvl::Image>(srcImage);
    auto dst_image = Get<vvl::Image>(dstImage);
    const ImageTransferCommand command = MakeImageBlitCommand(src_image.get(), dst_image.get(), regionCount, pRegions);
    return command.Validate(cb_context, loc);
}

bool SyncValidator::PreCallValidateCmdBlitImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
                                                VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount,
                                                const VkImageBlit* pRegions, VkFilter filter, const ErrorObject& error_obj) const {
    return ValidateCmdBlitImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter,
                                error_obj.location);
}

bool SyncValidator::PreCallValidateCmdBlitImage2KHR(VkCommandBuffer commandBuffer, const VkBlitImageInfo2KHR* pBlitImageInfo,
                                                    const ErrorObject& error_obj) const {
    return PreCallValidateCmdBlitImage2(commandBuffer, pBlitImageInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdBlitImage2(VkCommandBuffer commandBuffer, const VkBlitImageInfo2* pBlitImageInfo,
                                                 const ErrorObject& error_obj) const {
    return ValidateCmdBlitImage(commandBuffer, pBlitImageInfo->srcImage, pBlitImageInfo->srcImageLayout, pBlitImageInfo->dstImage,
                                pBlitImageInfo->dstImageLayout, pBlitImageInfo->regionCount, pBlitImageInfo->pRegions,
                                pBlitImageInfo->filter, error_obj.location.dot(Field::pBlitImageInfo));
}

bool SyncValidator::ValidateIndirectBuffer(const CommandBufferContext& cb_context, const AccessContext& access_context,
                                           const VkDeviceSize struct_size, const VkBuffer buffer, const VkDeviceSize offset,
                                           const uint32_t drawCount, const uint32_t stride, const Location& loc) const {
    bool skip = false;
    if (drawCount == 0) return skip;

    auto buf_state = Get<vvl::Buffer>(buffer);
    VkDeviceSize size = struct_size;
    if (drawCount == 1 || stride == size) {
        if (drawCount > 1) size *= drawCount;
        const AccessRange range = MakeRange(offset, size);
        auto hazard = access_context.DetectHazard(*buf_state, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, range);
        if (hazard.IsHazard()) {
            const LogObjectList objlist(cb_context.GetCBState().Handle(), buf_state->Handle());
            const std::string resource_description = "indirect " + FormatHandle(buffer);
            const auto error = error_messages_.BufferError(hazard, cb_context, loc.function, resource_description, range);
            skip |= SyncError(hazard.Hazard(), objlist, loc, error);
        }
    } else {
        for (uint32_t i = 0; i < drawCount; ++i) {
            const AccessRange range = MakeRange(offset + i * stride, size);
            auto hazard = access_context.DetectHazard(*buf_state, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, range);
            if (hazard.IsHazard()) {
                const LogObjectList objlist(cb_context.GetCBState().Handle(), buf_state->Handle());
                const std::string resource_description = "indirect " + FormatHandle(buffer);
                const auto error = error_messages_.BufferError(hazard, cb_context, loc.function, resource_description, range);
                skip |= SyncError(hazard.Hazard(), objlist, loc, error);
                break;
            }
        }
    }
    return skip;
}

ResourceAccessCommand SyncValidator::MakeIndirectBufferAccessCommand(VkDeviceSize struct_size, VkBuffer buffer,
                                                                     VkDeviceSize offset, uint32_t draw_count,
                                                                     uint32_t stride) const {
    ResourceAccessCommand command;
    if (draw_count == 0) return command;
    const auto buffer_state = Get<vvl::Buffer>(buffer);
    if (!buffer_state) return command;

    auto add_access = [&](AccessRange range) {
        ResourceAccessCommand::BufferAccess access;
        access.buffer = buffer_state.get();
        access.access_index = SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ;
        access.range = range;
        access.tag_handle = buffer_state->Handle();
        access.resource_name = "indirect ";
        command.accesses.emplace_back(std::move(access));
    };
    VkDeviceSize size = struct_size;
    if (draw_count == 1 || stride == size) {
        if (draw_count > 1) size *= draw_count;
        add_access(MakeRange(offset, size));
    } else {
        for (uint32_t i = 0; i < draw_count; ++i) add_access(MakeRange(offset + i * stride, size));
    }
    return command;
}

void SyncValidator::RecordIndirectBuffer(CommandBufferContext& cb_context, const ResourceUsageTag tag,
                                         const VkDeviceSize struct_size, const VkBuffer buffer, const VkDeviceSize offset,
                                         const uint32_t drawCount, uint32_t stride) {
    auto buf_state = Get<vvl::Buffer>(buffer);
    auto tag_ex = buf_state ? cb_context.AddCommandHandle(tag, buf_state->Handle()) : ResourceUsageTagEx{tag};

    VkDeviceSize size = struct_size;
    AccessContext& access_context = cb_context.GetCurrentAccessContext();
    if (drawCount == 1 || stride == size) {
        if (drawCount > 1) size *= drawCount;
        const AccessRange range = MakeRange(offset, size);
        access_context.UpdateAccessState(*buf_state, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, range, tag_ex);
    } else {
        for (uint32_t i = 0; i < drawCount; ++i) {
            const AccessRange range = MakeRange(offset + i * stride, size);
            access_context.UpdateAccessState(*buf_state, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, range, tag_ex);
        }
    }
}

bool SyncValidator::ValidateCountBuffer(const CommandBufferContext& cb_context, const AccessContext& access_context,
                                        VkBuffer buffer, VkDeviceSize offset, const Location& loc,
                                        const char* count_buffer_label) const {
    bool skip = false;

    auto count_buf_state = Get<vvl::Buffer>(buffer);
    const AccessRange range = MakeRange(offset, 4);
    auto hazard = access_context.DetectHazard(*count_buf_state, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, range);
    if (hazard.IsHazard()) {
        const LogObjectList objlist(cb_context.GetCBState().Handle(), count_buf_state->Handle());
        const std::string resource_description = std::string(count_buffer_label) + " " + FormatHandle(buffer);
        const auto error = error_messages_.BufferError(hazard, cb_context, loc.function, resource_description, range);
        skip |= SyncError(hazard.Hazard(), objlist, loc, error);
    }
    return skip;
}

ResourceAccessCommand SyncValidator::MakeCountBufferAccessCommand(VkBuffer buffer, VkDeviceSize offset,
                                                                  const char* label) const {
    ResourceAccessCommand command;
    const auto buffer_state = Get<vvl::Buffer>(buffer);
    if (!buffer_state) return command;
    ResourceAccessCommand::BufferAccess access;
    access.buffer = buffer_state.get();
    access.access_index = SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ;
    access.range = MakeRange(offset, 4);
    access.tag_handle = buffer_state->Handle();
    access.resource_name = std::string(label) + " ";
    command.accesses.emplace_back(std::move(access));
    return command;
}

void SyncValidator::RecordCountBuffer(CommandBufferContext& cb_context, const ResourceUsageTag tag, VkBuffer buffer,
                                      VkDeviceSize offset) {
    auto count_buf_state = Get<vvl::Buffer>(buffer);
    const AccessRange range = MakeRange(offset, 4);
    const ResourceUsageTagEx tag_ex = cb_context.AddCommandHandle(tag, count_buf_state->Handle());
    AccessContext& access_context = cb_context.GetCurrentAccessContext();
    access_context.UpdateAccessState(*count_buf_state, SYNC_DRAW_INDIRECT_INDIRECT_COMMAND_READ, range, tag_ex);
}

bool SyncValidator::PreCallValidateCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z,
                                               const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    return cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_COMPUTE)
        .Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z,
                                              const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    cb_context.RecordResourceAccesses(tag, cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_COMPUTE),
                                      syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                       const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_COMPUTE);
    command.Append(
        MakeIndirectBufferAccessCommand(sizeof(VkDispatchIndirectCommand), buffer, offset, 1, sizeof(VkDispatchIndirectCommand)));
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                      const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_COMPUTE);
    command.Append(
        MakeIndirectBufferAccessCommand(sizeof(VkDispatchIndirectCommand), buffer, offset, 1, sizeof(VkDispatchIndirectCommand)));
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX, uint32_t baseGroupY,
                                                   uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY,
                                                   uint32_t groupCountZ, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    return cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_COMPUTE)
        .Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdDispatchBaseKHR(VkCommandBuffer commandBuffer, uint32_t baseGroupX, uint32_t baseGroupY,
                                                      uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY,
                                                      uint32_t groupCountZ, const ErrorObject& error_obj) const {
    return PreCallValidateCmdDispatchBase(commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ,
                                          error_obj);
}

void SyncValidator::PostCallRecordCmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX, uint32_t baseGroupY,
                                                  uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY,
                                                  uint32_t groupCountZ, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    cb_context.RecordResourceAccesses(tag, cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_COMPUTE),
                                      syncval_settings.IsRecordTimeValidationEnabled());
}

void SyncValidator::PostCallRecordCmdDispatchBaseKHR(VkCommandBuffer commandBuffer, uint32_t baseGroupX, uint32_t baseGroupY,
                                                     uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY,
                                                     uint32_t groupCountZ, const RecordObject& record_obj) {
    PostCallRecordCmdDispatchBase(commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ,
                                  record_obj);
}

bool SyncValidator::PreCallValidateCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                                           uint32_t firstVertex, uint32_t firstInstance, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawVertexAccessCommand(vertexCount, firstVertex));
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                                          uint32_t firstVertex, uint32_t firstInstance, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawVertexAccessCommand(vertexCount, firstVertex));
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                                                  uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance,
                                                  const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawVertexIndexAccessCommand(indexCount, firstIndex));
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                                                 uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance,
                                                 const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawVertexIndexAccessCommand(indexCount, firstIndex));
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                   uint32_t drawCount, uint32_t stride, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled() || drawCount == 0) return false;

    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(MakeIndirectBufferAccessCommand(sizeof(VkDrawIndirectCommand), buffer, offset, drawCount, stride));
    // TODO: Shader instrumentation support is needed to read indirect buffer content (new syncval mode)
    // skip |= cb_context->ValidateDrawVertex(?, ?, error_obj.location);
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                  uint32_t drawCount, uint32_t stride, const RecordObject& record_obj) {
    if (drawCount == 0) {
        return;
    }
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(MakeIndirectBufferAccessCommand(sizeof(VkDrawIndirectCommand), buffer, offset, drawCount, stride));
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());

    // TODO: Shader instrumentation support is needed to read indirect buffer content (new syncval mode)
    // cb_context->RecordDrawVertex(?, ?, tag);
}

bool SyncValidator::PreCallValidateCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                          uint32_t drawCount, uint32_t stride, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled() || drawCount == 0) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(MakeIndirectBufferAccessCommand(sizeof(VkDrawIndexedIndirectCommand), buffer, offset, drawCount, stride));

    // TODO: Shader instrumentation support is needed to read indirect buffer content (new syncval mode)
    // skip |= cb_context->ValidateDrawVertexIndex(?, ?, error_obj.location);
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                         uint32_t drawCount, uint32_t stride, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(MakeIndirectBufferAccessCommand(sizeof(VkDrawIndexedIndirectCommand), buffer, offset, drawCount, stride));
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());

    // TODO: Shader instrumentation support is needed to read indirect buffer content (new syncval mode)
    // cb_context->RecordDrawVertexIndex(?, ?, tag);
}

bool SyncValidator::PreCallValidateCmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                        VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                        uint32_t stride, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(MakeIndirectBufferAccessCommand(sizeof(VkDrawIndirectCommand), buffer, offset, maxDrawCount, stride));
    command.Append(MakeCountBufferAccessCommand(countBuffer, countBufferOffset));

    // TODO: Shader instrumentation support is needed to read indirect buffer content (new syncval mode)
    // skip |= cb_context->ValidateDrawVertex(?, ?, error_obj.location);
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::RecordCmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                               VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                               uint32_t stride, Func command) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(command);

    auto recorded_command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    recorded_command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    recorded_command.Append(
        MakeIndirectBufferAccessCommand(sizeof(VkDrawIndirectCommand), buffer, offset, maxDrawCount, stride));
    recorded_command.Append(MakeCountBufferAccessCommand(countBuffer, countBufferOffset));
    cb_context.RecordResourceAccesses(tag, std::move(recorded_command), syncval_settings.IsRecordTimeValidationEnabled());

    // TODO: Shader instrumentation support is needed to read indirect buffer content (new syncval mode)
    // cb_context->RecordDrawVertex(?, ?, tag);
}

void SyncValidator::PostCallRecordCmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                       VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                       uint32_t stride, const RecordObject& record_obj) {
    RecordCmdDrawIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride,
                               record_obj.location.function);
}
bool SyncValidator::PreCallValidateCmdDrawIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                           VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                                           uint32_t maxDrawCount, uint32_t stride,
                                                           const ErrorObject& error_obj) const {
    return PreCallValidateCmdDrawIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride,
                                               error_obj);
}

void SyncValidator::PostCallRecordCmdDrawIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                          VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                                          uint32_t maxDrawCount, uint32_t stride, const RecordObject& record_obj) {
    PostCallRecordCmdDrawIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride,
                                       record_obj);
}

bool SyncValidator::PreCallValidateCmdDrawIndirectCountAMD(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                           VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                                           uint32_t maxDrawCount, uint32_t stride,
                                                           const ErrorObject& error_obj) const {
    return PreCallValidateCmdDrawIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride,
                                               error_obj);
}

void SyncValidator::PostCallRecordCmdDrawIndirectCountAMD(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                          VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                                          uint32_t maxDrawCount, uint32_t stride, const RecordObject& record_obj) {
    PostCallRecordCmdDrawIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride,
                                       record_obj);
}

bool SyncValidator::PreCallValidateCmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                               VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                                               uint32_t maxDrawCount, uint32_t stride,
                                                               const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(
        MakeIndirectBufferAccessCommand(sizeof(VkDrawIndexedIndirectCommand), buffer, offset, maxDrawCount, stride));
    command.Append(MakeCountBufferAccessCommand(countBuffer, countBufferOffset));

    // TODO: Shader instrumentation support is needed to read indirect buffer content (new syncval mode)
    // skip |= cb_context->ValidateDrawVertexIndex(?, ?, error_obj.location);
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::RecordCmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                      VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                      uint32_t stride, Func command) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(command);

    auto recorded_command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    recorded_command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    recorded_command.Append(
        MakeIndirectBufferAccessCommand(sizeof(VkDrawIndexedIndirectCommand), buffer, offset, maxDrawCount, stride));
    recorded_command.Append(MakeCountBufferAccessCommand(countBuffer, countBufferOffset));
    cb_context.RecordResourceAccesses(tag, std::move(recorded_command), syncval_settings.IsRecordTimeValidationEnabled());

    // TODO: Shader instrumentation support is needed to read indirect buffer content (new syncval mode)
    // cb_context->RecordDrawVertexIndex(?, ?, tag);
}

void SyncValidator::PostCallRecordCmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                              VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                                              uint32_t maxDrawCount, uint32_t stride,
                                                              const RecordObject& record_obj) {
    RecordCmdDrawIndexedIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride,
                                      record_obj.location.function);
}

bool SyncValidator::PreCallValidateCmdDrawIndexedIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                                                  VkDeviceSize offset, VkBuffer countBuffer,
                                                                  VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                                  uint32_t stride, const ErrorObject& error_obj) const {
    return PreCallValidateCmdDrawIndexedIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount,
                                                      stride, error_obj);
}

void SyncValidator::PostCallRecordCmdDrawIndexedIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                                                 VkDeviceSize offset, VkBuffer countBuffer,
                                                                 VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                                 uint32_t stride, const RecordObject& record_obj) {
    PostCallRecordCmdDrawIndexedIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride,
                                              record_obj);
}

bool SyncValidator::PreCallValidateCmdDrawIndexedIndirectCountAMD(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                                                  VkDeviceSize offset, VkBuffer countBuffer,
                                                                  VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                                  uint32_t stride, const ErrorObject& error_obj) const {
    return PreCallValidateCmdDrawIndexedIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount,
                                                      stride, error_obj);
}

void SyncValidator::PostCallRecordCmdDrawIndexedIndirectCountAMD(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                                                 VkDeviceSize offset, VkBuffer countBuffer,
                                                                 VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                                 uint32_t stride, const RecordObject& record_obj) {
    PostCallRecordCmdDrawIndexedIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride,
                                              record_obj);
}

bool SyncValidator::PreCallValidateCmdDrawMeshTasksEXT(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY,
                                                       uint32_t groupCountZ, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDrawMeshTasksEXT(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY,
                                                      uint32_t groupCountZ, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdDrawMeshTasksIndirectEXT(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                               uint32_t drawCount, uint32_t stride,
                                                               const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled() || drawCount == 0) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(
        MakeIndirectBufferAccessCommand(sizeof(VkDrawMeshTasksIndirectCommandEXT), buffer, offset, drawCount, stride));
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDrawMeshTasksIndirectEXT(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                              uint32_t drawCount, uint32_t stride, const RecordObject& record_obj) {
    if (drawCount == 0) {
        return;
    }
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(
        MakeIndirectBufferAccessCommand(sizeof(VkDrawMeshTasksIndirectCommandEXT), buffer, offset, drawCount, stride));
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                                                    VkDeviceSize offset, VkBuffer countBuffer,
                                                                    VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                                    uint32_t stride, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) return false;

    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(MakeCountBufferAccessCommand(countBuffer, countBufferOffset));
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                                                   VkDeviceSize offset, VkBuffer countBuffer,
                                                                   VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                                   uint32_t stride, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(MakeCountBufferAccessCommand(countBuffer, countBufferOffset));
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                                                          const VkMultiDrawIndexedInfoEXT* pIndexInfo, uint32_t instanceCount,
                                                          uint32_t firstInstance, uint32_t stride, const int32_t* pVertexOffset,
                                                          const ErrorObject& error_obj) const {
    if (!pIndexInfo || !syncval_settings.IsRecordTimeValidationEnabled()) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    const auto ptr = reinterpret_cast<const uint8_t*>(pIndexInfo);
    for (uint32_t i = 0; i < drawCount; i++) {
        const auto info_ptr = reinterpret_cast<const VkMultiDrawIndexedInfoEXT*>(ptr + i * stride);
        command.Append(cb_context.MakeDrawVertexIndexAccessCommand(info_ptr->indexCount, info_ptr->firstIndex));
    }
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                                                         const VkMultiDrawIndexedInfoEXT* pIndexInfo, uint32_t instanceCount,
                                                         uint32_t firstInstance, uint32_t stride, const int32_t* pVertexOffset,
                                                         const RecordObject& record_obj) {
    if (!pIndexInfo) {
        return;
    }
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    const auto ptr = reinterpret_cast<const uint8_t*>(pIndexInfo);
    for (uint32_t i = 0; i < drawCount; i++) {
        const auto info_ptr = reinterpret_cast<const VkMultiDrawIndexedInfoEXT*>(ptr + i * stride);
        command.Append(cb_context.MakeDrawVertexIndexAccessCommand(info_ptr->indexCount, info_ptr->firstIndex));
    }
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                                                   const VkMultiDrawInfoEXT* pVertexInfo, uint32_t instanceCount,
                                                   uint32_t firstInstance, uint32_t stride, const ErrorObject& error_obj) const {
    if (!pVertexInfo || !syncval_settings.IsRecordTimeValidationEnabled()) return false;
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    const auto ptr = reinterpret_cast<const uint8_t*>(pVertexInfo);
    for (uint32_t i = 0; i < drawCount; i++) {
        const auto info_ptr = reinterpret_cast<const VkMultiDrawInfoEXT*>(ptr + i * stride);
        command.Append(cb_context.MakeDrawVertexAccessCommand(info_ptr->vertexCount, info_ptr->firstVertex));
    }
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                                                  const VkMultiDrawInfoEXT* pVertexInfo, uint32_t instanceCount,
                                                  uint32_t firstInstance, uint32_t stride, const RecordObject& record_obj) {
    if (!pVertexInfo) {
        return;
    }
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    const auto ptr = reinterpret_cast<const uint8_t*>(pVertexInfo);
    for (uint32_t i = 0; i < drawCount; i++) {
        const auto info_ptr = reinterpret_cast<const VkMultiDrawInfoEXT*>(ptr + i * stride);
        command.Append(cb_context.MakeDrawVertexAccessCommand(info_ptr->vertexCount, info_ptr->firstVertex));
    }
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer, uint32_t instanceCount,
                                                               uint32_t firstInstance, VkBuffer counterBuffer,
                                                               VkDeviceSize counterBufferOffset, uint32_t counterOffset,
                                                               uint32_t vertexStride, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) return false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(MakeCountBufferAccessCommand(counterBuffer, counterBufferOffset, "transform feedback counter"));
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer, uint32_t instanceCount,
                                                              uint32_t firstInstance, VkBuffer counterBuffer,
                                                              VkDeviceSize counterBufferOffset, uint32_t counterOffset,
                                                              uint32_t vertexStride, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);

    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_GRAPHICS);
    command.Append(cb_context.MakeDrawAttachmentAccessCommand());
    command.Append(MakeCountBufferAccessCommand(counterBuffer, counterBufferOffset, "transform feedback counter"));
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdClearColorImage(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout,
                                                      const VkClearColorValue* pColor, uint32_t rangeCount,
                                                      const VkImageSubresourceRange* pRanges, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto image_state = Get<vvl::Image>(image);
    const ImageTransferCommand command = MakeImageClearCommand(image_state.get(), rangeCount, pRanges);
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage image,
                                                             VkImageLayout imageLayout,
                                                             const VkClearDepthStencilValue* pDepthStencil, uint32_t rangeCount,
                                                             const VkImageSubresourceRange* pRanges,
                                                             const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto image_state = Get<vvl::Image>(image);
    const ImageTransferCommand command = MakeImageClearCommand(image_state.get(), rangeCount, pRanges);
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                                                       const VkClearAttachment* pAttachments, uint32_t rectCount,
                                                       const VkClearRect* pRects, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    ResourceAccessCommand command;
    for (const VkClearAttachment& attachment : vvl::make_span(pAttachments, attachmentCount)) {
        for (const auto [rect_index, rect] : vvl::enumerate(pRects, rectCount)) {
            command.Append(cb_context.MakeClearAttachmentAccessCommand(attachment, rect_index, rect));
        }
    }
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                                                           uint32_t firstQuery, uint32_t queryCount, VkBuffer dstBuffer,
                                                           VkDeviceSize dstOffset, VkDeviceSize stride, VkQueryResultFlags flags,
                                                           const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled() || queryCount == 0) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto dst_buffer = Get<vvl::Buffer>(dstBuffer);
    if (!dst_buffer) {
        return false;
    }
    const uint32_t query_size = (flags & VK_QUERY_RESULT_64_BIT) ? 8 : 4;
    const VkDeviceSize range_size = (queryCount - 1) * stride + query_size;
    const BufferAccessCommand command{*dst_buffer, SYNC_COPY_TRANSFER_WRITE, MakeRange(dstOffset, range_size), 0, queryPool,
                                      "dstBuffer "};
    // TODO:Track VkQueryPool
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                                 VkDeviceSize size, uint32_t data, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto dst_buffer = Get<vvl::Buffer>(dstBuffer);
    if (!dst_buffer) {
        return false;
    }
    const BufferAccessCommand command{*dst_buffer, SYNC_CLEAR_TRANSFER_WRITE, MakeRange(*dst_buffer, dstOffset, size), 0,
                                      VK_NULL_HANDLE, "dstBuffer "};
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdResolveImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
                                                   VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount,
                                                   const VkImageResolve* pRegions, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto src_image = Get<vvl::Image>(srcImage);
    auto dst_image = Get<vvl::Image>(dstImage);
    const ImageTransferCommand command = MakeImageResolveCommand(src_image.get(), dst_image.get(), regionCount, pRegions);
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdResolveImage2(VkCommandBuffer commandBuffer, const VkResolveImageInfo2* pResolveImageInfo,
                                                    const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled() || !pResolveImageInfo) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto src_image = Get<vvl::Image>(pResolveImageInfo->srcImage);
    auto dst_image = Get<vvl::Image>(pResolveImageInfo->dstImage);
    const ImageTransferCommand command = MakeImageResolveCommand(src_image.get(), dst_image.get(), pResolveImageInfo->regionCount,
                                                                 pResolveImageInfo->pRegions);
    return command.Validate(cb_context, error_obj.location.dot(Field::pResolveImageInfo));
}

bool SyncValidator::PreCallValidateCmdResolveImage2KHR(VkCommandBuffer commandBuffer,
                                                       const VkResolveImageInfo2KHR* pResolveImageInfo,
                                                       const ErrorObject& error_obj) const {
    return PreCallValidateCmdResolveImage2(commandBuffer, pResolveImageInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                                   VkDeviceSize dataSize, const void* pData, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto dst_buffer = Get<vvl::Buffer>(dstBuffer);
    if (!dst_buffer) {
        return false;
    }
    // VK_WHOLE_SIZE not allowed
    const BufferAccessCommand command{*dst_buffer, SYNC_CLEAR_TRANSFER_WRITE, MakeRange(dstOffset, dataSize), 0,
                                      VK_NULL_HANDLE, "dstBuffer "};
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdWriteBufferMarkerAMD(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage,
                                                           VkBuffer dstBuffer, VkDeviceSize dstOffset, uint32_t marker,
                                                           const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto dst_buffer = Get<vvl::Buffer>(dstBuffer);
    if (!dst_buffer) {
        return false;
    }
    const BufferAccessCommand command{*dst_buffer, SYNC_COPY_TRANSFER_WRITE, MakeRange(dstOffset, 4), SyncFlag::kMarker,
                                      VK_NULL_HANDLE, "dstBuffer "};
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdWriteBufferMarkerAMD(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage,
                                                          VkBuffer dstBuffer, VkDeviceSize dstOffset, uint32_t marker,
                                                          const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    auto dst_buffer = Get<vvl::Buffer>(dstBuffer);
    if (!dst_buffer) {
        return;
    }
    const ResourceUsageTagEx tag_ex = cb_context.AddCommandHandle(tag, dst_buffer->Handle());
    const BufferAccessCommand command{*dst_buffer, SYNC_COPY_TRANSFER_WRITE, MakeRange(dstOffset, 4), SyncFlag::kMarker,
                                      VK_NULL_HANDLE, "dstBuffer ", tag_ex.handle_index};
    if (syncval_settings.IsRecordTimeValidationEnabled()) {
        command.Apply(cb_context.GetSyncEnvironment(), tag, cb_context.GetCurrentAccessContext());
    }
    if (syncval_settings.full_validation) {
        cb_context.StoreCommand(tag, command);
    }
}

ResourceAccessCommand SyncValidator::MakeDecodeVideoAccessCommand(const vvl::CommandBuffer& cb_state,
                                                                  const VkVideoDecodeInfoKHR& decode_info) const {
    ResourceAccessCommand command;
    const auto vs_state = cb_state.bound_video_session;
    if (!vs_state) {
        return command;
    }

    if (auto src_buffer = Get<vvl::Buffer>(decode_info.srcBuffer)) {
        ResourceAccessCommand::BufferAccess access;
        access.buffer = src_buffer.get();
        access.access_index = SYNC_VIDEO_DECODE_VIDEO_DECODE_READ;
        access.range = MakeRange(*access.buffer, decode_info.srcBufferOffset, decode_info.srcBufferRange);
        access.legacy_record_time_object_only = true;
        access.tag_handle = access.buffer->Handle();
        access.resource_name = "bitstream buffer ";
        command.accesses.emplace_back(std::move(access));
    }

    auto add_picture = [&](const vvl::VideoPictureResource& resource, SyncAccessIndex access_index,
                           std::string resource_description) {
        if (!resource) {
            return;
        }
        ResourceAccessCommand::ImageRangeAccess access;
        access.image = resource.image_state.get();
        access.access_index = access_index;
        access.subresource_range = resource.range;
        access.use_offset_extent = true;
        access.offset = resource.GetEffectiveImageOffset(*vs_state);
        access.extent = resource.GetEffectiveImageExtent(*vs_state);
        access.tag_handle = resource.image_view_state->Handle();
        access.legacy_register_handle = true;
        access.resource_description = std::move(resource_description);
        access.error_type = ResourceAccessCommand::ImageRangeAccess::ErrorType::kVideo;
        command.accesses.emplace_back(std::move(access));
    };

    const auto dst_resource = vvl::VideoPictureResource(*device_state, decode_info.dstPictureResource);
    if (dst_resource) {
        std::ostringstream ss;
        ss << "decode output picture ";
        ss << Location(Func::Empty, Field::pDecodeInfo).dot(Field::dstPictureResource).Fields() << " ";
        FormatVideoPictureResouce(*this, decode_info.dstPictureResource, ss);
        add_picture(dst_resource, SYNC_VIDEO_DECODE_VIDEO_DECODE_WRITE, ss.str());
    }

    if (decode_info.pSetupReferenceSlot && decode_info.pSetupReferenceSlot->pPictureResource) {
        const auto& video_picture = *decode_info.pSetupReferenceSlot->pPictureResource;
        const auto setup_resource = vvl::VideoPictureResource(*device_state, video_picture);
        if (setup_resource && setup_resource != dst_resource) {
            std::ostringstream ss;
            ss << "reconstructed picture ";
            ss << Location(Func::Empty, Field::pDecodeInfo).dot(Field::pSetupReferenceSlot).dot(Field::pPictureResource).Fields()
               << " ";
            FormatVideoPictureResouce(*this, video_picture, ss);
            add_picture(setup_resource, SYNC_VIDEO_DECODE_VIDEO_DECODE_WRITE, ss.str());
        }
    }

    for (uint32_t i = 0; i < decode_info.referenceSlotCount; ++i) {
        if (!decode_info.pReferenceSlots[i].pPictureResource) {
            continue;
        }
        const auto& video_picture = *decode_info.pReferenceSlots[i].pPictureResource;
        const auto reference_resource = vvl::VideoPictureResource(*device_state, video_picture);
        if (reference_resource) {
            std::ostringstream ss;
            ss << "reference picture " << i << " ";
            ss << Location(Func::Empty, Field::pDecodeInfo).dot(Field::pReferenceSlots, i).dot(Field::pPictureResource).Fields()
               << " ";
            FormatVideoPictureResouce(*this, video_picture, ss);
            add_picture(reference_resource, SYNC_VIDEO_DECODE_VIDEO_DECODE_READ, ss.str());
        }
    }
    return command;
}

ResourceAccessCommand SyncValidator::MakeEncodeVideoAccessCommand(const vvl::CommandBuffer& cb_state,
                                                                  const VkVideoEncodeInfoKHR& encode_info) const {
    ResourceAccessCommand command;
    const auto vs_state = cb_state.bound_video_session;
    if (!vs_state) {
        return command;
    }

    if (auto dst_buffer = Get<vvl::Buffer>(encode_info.dstBuffer)) {
        ResourceAccessCommand::BufferAccess access;
        access.buffer = dst_buffer.get();
        access.access_index = SYNC_VIDEO_ENCODE_VIDEO_ENCODE_WRITE;
        access.range = MakeRange(*access.buffer, encode_info.dstBufferOffset, encode_info.dstBufferRange);
        access.legacy_record_time_object_only = true;
        access.tag_handle = access.buffer->Handle();
        access.resource_name = "bitstream buffer ";
        command.accesses.emplace_back(std::move(access));
    }

    auto add_picture = [&](const vvl::VideoPictureResource& resource, SyncAccessIndex access_index,
                           std::string resource_description) {
        if (!resource) {
            return;
        }
        ResourceAccessCommand::ImageRangeAccess access;
        access.image = resource.image_state.get();
        access.access_index = access_index;
        access.subresource_range = resource.range;
        access.use_offset_extent = true;
        access.offset = resource.GetEffectiveImageOffset(*vs_state);
        access.extent = resource.GetEffectiveImageExtent(*vs_state);
        access.tag_handle = resource.image_view_state->Handle();
        access.legacy_register_handle = true;
        access.resource_description = std::move(resource_description);
        access.error_type = ResourceAccessCommand::ImageRangeAccess::ErrorType::kVideo;
        command.accesses.emplace_back(std::move(access));
    };

    const auto src_resource = vvl::VideoPictureResource(*device_state, encode_info.srcPictureResource);
    if (src_resource) {
        std::ostringstream ss;
        ss << "encode input picture ";
        ss << Location(Func::Empty, Field::pEncodeInfo).dot(Field::srcPictureResource).Fields() << " ";
        FormatVideoPictureResouce(*this, encode_info.srcPictureResource, ss);
        add_picture(src_resource, SYNC_VIDEO_ENCODE_VIDEO_ENCODE_READ, ss.str());
    }

    if (encode_info.pSetupReferenceSlot && encode_info.pSetupReferenceSlot->pPictureResource) {
        const auto& video_picture = *encode_info.pSetupReferenceSlot->pPictureResource;
        const auto setup_resource = vvl::VideoPictureResource(*device_state, video_picture);
        if (setup_resource) {
            std::ostringstream ss;
            ss << "reconstructed picture ";
            ss << Location(Func::Empty, Field::pEncodeInfo).dot(Field::pSetupReferenceSlot).dot(Field::pPictureResource).Fields()
               << " ";
            FormatVideoPictureResouce(*this, video_picture, ss);
            add_picture(setup_resource, SYNC_VIDEO_ENCODE_VIDEO_ENCODE_WRITE, ss.str());
        }
    }

    for (uint32_t i = 0; i < encode_info.referenceSlotCount; ++i) {
        if (!encode_info.pReferenceSlots[i].pPictureResource) {
            continue;
        }
        const auto& video_picture = *encode_info.pReferenceSlots[i].pPictureResource;
        const auto reference_resource = vvl::VideoPictureResource(*device_state, video_picture);
        if (reference_resource) {
            std::ostringstream ss;
            ss << "reference picture " << i << " ";
            ss << Location(Func::Empty, Field::pEncodeInfo).dot(Field::pReferenceSlots, i).dot(Field::pPictureResource).Fields()
               << " ";
            FormatVideoPictureResouce(*this, video_picture, ss);
            add_picture(reference_resource, SYNC_VIDEO_ENCODE_VIDEO_ENCODE_READ, ss.str());
        }
    }

    if (encode_info.flags & (VK_VIDEO_ENCODE_WITH_QUANTIZATION_DELTA_MAP_BIT_KHR | VK_VIDEO_ENCODE_WITH_EMPHASIS_MAP_BIT_KHR)) {
        const auto quantization_map_info = vku::FindStructInPNextChain<VkVideoEncodeQuantizationMapInfoKHR>(encode_info.pNext);
        if (quantization_map_info) {
            auto image_view = Get<vvl::ImageView>(quantization_map_info->quantizationMap);
            if (image_view) {
                ResourceAccessCommand::ImageRangeAccess access;
                access.image = image_view->image_state.get();
                access.access_index = SYNC_VIDEO_ENCODE_VIDEO_ENCODE_READ;
                access.subresource_range = image_view->normalized_subresource_range;
                access.is_depth_sliced = image_view->is_depth_sliced;
                access.use_offset_extent = true;
                access.extent = {quantization_map_info->quantizationMapExtent.width,
                                 quantization_map_info->quantizationMapExtent.height, 1};
                access.tag_handle = image_view->Handle();
                access.legacy_register_handle = true;
                std::ostringstream ss;
                ss << "quantization map ";
                ss << Location(Func::Empty, Field::pEncodeInfo).dot(Field::quantizationMap).Fields() << " ";
                FormatVideoQuantizationMap(*this, *quantization_map_info, ss);
                access.resource_description = ss.str();
                access.error_type = ResourceAccessCommand::ImageRangeAccess::ErrorType::kVideo;
                command.accesses.emplace_back(std::move(access));
            }
        }
    }
    return command;
}

bool SyncValidator::PreCallValidateCmdDecodeVideoKHR(VkCommandBuffer commandBuffer, const VkVideoDecodeInfoKHR* pDecodeInfo,
                                                     const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    return MakeDecodeVideoAccessCommand(*cb_state, *pDecodeInfo).Validate(GetCommandBufferContext(*cb_state), error_obj.location);
}
bool SyncValidator::PreCallValidateCmdEncodeVideoKHR(VkCommandBuffer commandBuffer, const VkVideoEncodeInfoKHR* pEncodeInfo,
                                                     const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    return MakeEncodeVideoAccessCommand(*cb_state, *pEncodeInfo).Validate(GetCommandBufferContext(*cb_state), error_obj.location);
}
void SyncValidator::PostCallRecordResetEvent(VkDevice device, VkEvent event, const RecordObject& record_obj) {
    if (record_obj.result != VK_SUCCESS) {
        return;
    }
    if (auto event_state = Get<vvl::Event>(event)) {
        const auto all_batches = GetAllQueueBatchContexts();
        for (const auto& batch : all_batches) {
            SyncEventsContext& events_context = batch->GetEventsContext();
            if (SyncEventState* sync_event = events_context.GetFromShared(event_state)) {
                sync_event->last_command = record_obj.location.function;
                sync_event->unsynchronized_set = vvl::Func::Empty;
                sync_event->ResetFirstScope();
                sync_event->barriers = 0;
            }
        }
    }
}

bool SyncValidator::PreCallValidateCmdSetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask,
                                               const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    EventCommand command;
    command.type = EventCommand::Type::kSet;
    command.events.emplace_back(Get<vvl::Event>(event));
    command.exec_scope = SyncExecScope::MakeSrc(cb_state->GetQueueFlags(), stageMask);
    command.command = error_obj.location.function;
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::RecordCmdSetEvent(CommandBufferContext& cb_context, std::shared_ptr<const vvl::Event>&& event,
                                      const SyncExecScope& src_exec_scope, const Location& loc) const {
    const ResourceUsageTag tag = cb_context.NextCommandTag(loc.function);

    EventCommand command;
    command.type = EventCommand::Type::kSet;
    command.events.emplace_back(event);
    command.exec_scope = src_exec_scope;
    command.command = loc.function;
    if (syncval_settings.IsRecordTimeValidationEnabled()) {
        command.Apply(cb_context.GetSyncEnvironment(), tag, cb_context.GetCurrentAccessContext());
    }
    cb_context.StoreCommand(tag, command);

    // Snapshot the current access_context for later inspection at wait time.
    // NOTE: This appears brute force, but given that we only save a "first-last" model
    // of access history, the current access context (include barrier state for chaining)
    // won't necessarily contain the needed information at Wait or Submit time reference.
    auto src_access_context = std::make_shared<AccessContext>(*this);
    src_access_context->InitFrom(cb_context.GetCbAccessContext());

    cb_context.AddReplayEntry(tag, false, SetEventReplay(std::move(event), src_exec_scope, std::move(src_access_context), loc));
}

void SyncValidator::PostCallRecordCmdSetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask,
                                              const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    auto event_state = Get<vvl::Event>(event);

    const SyncExecScope src_exec_scope = SyncExecScope::MakeSrc(queue_flags, stageMask);
    RecordCmdSetEvent(cb_context, std::move(event_state), src_exec_scope, record_obj.location);
}

bool SyncValidator::PreCallValidateCmdSetEvent2KHR(VkCommandBuffer commandBuffer, VkEvent event,
                                                   const VkDependencyInfoKHR* pDependencyInfo, const ErrorObject& error_obj) const {
    return PreCallValidateCmdSetEvent2(commandBuffer, event, pDependencyInfo, error_obj);
}

bool SyncValidator::PreCallValidateCmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent event,
                                                const VkDependencyInfo* pDependencyInfo, const ErrorObject& error_obj) const {
    bool skip = false;
    if (!pDependencyInfo || !syncval_settings.IsRecordTimeValidationEnabled()) {
        return skip;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    EventCommand command;
    command.type = EventCommand::Type::kSet;
    command.events.emplace_back(Get<vvl::Event>(event));
    command.exec_scope = SyncExecScope::MakeSrc(queue_flags, sync_utils::GetExecScopes(*pDependencyInfo).src);
    command.command = error_obj.location.function;
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdSetEvent2KHR(VkCommandBuffer commandBuffer, VkEvent event,
                                                  const VkDependencyInfoKHR* pDependencyInfo, const RecordObject& record_obj) {
    PostCallRecordCmdSetEvent2(commandBuffer, event, pDependencyInfo, record_obj);
}

void SyncValidator::PostCallRecordCmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent event,
                                               const VkDependencyInfo* pDependencyInfo, const RecordObject& record_obj) {
    if (!pDependencyInfo) {
        return;
    }
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    auto event_state = Get<vvl::Event>(event);

    const SyncExecScope src_exec_scope = SyncExecScope::MakeSrc(queue_flags, sync_utils::GetExecScopes(*pDependencyInfo).src);
    RecordCmdSetEvent(cb_context, std::move(event_state), src_exec_scope, record_obj.location);
}

bool SyncValidator::PreCallValidateCmdResetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask,
                                                 const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    EventCommand command;
    command.type = EventCommand::Type::kReset;
    command.events.emplace_back(Get<vvl::Event>(event));
    command.exec_scope = SyncExecScope::MakeSrc(cb_state->GetQueueFlags(), stageMask);
    command.command = error_obj.location.function;
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::RecordCmdResetEvent(CommandBufferContext& cb_context, std::shared_ptr<const vvl::Event>&& event,
                                        const SyncExecScope& exec_scope, const Location& loc) const {
    const ResourceUsageTag tag = cb_context.NextCommandTag(loc.function);
    EventCommand command;
    command.type = EventCommand::Type::kReset;
    command.events.emplace_back(event);
    command.exec_scope = exec_scope;
    command.command = loc.function;
    if (syncval_settings.IsRecordTimeValidationEnabled()) {
        command.Apply(cb_context.GetSyncEnvironment(), tag, cb_context.GetCurrentAccessContext());
    }
    cb_context.StoreCommand(tag, command);
    cb_context.AddReplayEntry(tag, false, ResetEventReplay(std::move(event), exec_scope, loc));
}

void SyncValidator::PostCallRecordCmdResetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask,
                                                const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    auto event_state = Get<vvl::Event>(event);

    const SyncExecScope exec_scope = SyncExecScope::MakeSrc(queue_flags, stageMask);
    RecordCmdResetEvent(cb_context, std::move(event_state), exec_scope, record_obj.location);
}

bool SyncValidator::PreCallValidateCmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags2 stageMask,
                                                  const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    EventCommand command;
    command.type = EventCommand::Type::kReset;
    command.events.emplace_back(Get<vvl::Event>(event));
    command.exec_scope = SyncExecScope::MakeSrc(cb_state->GetQueueFlags(), stageMask);
    command.command = error_obj.location.function;
    return command.Validate(cb_context, error_obj.location);
}

bool SyncValidator::PreCallValidateCmdResetEvent2KHR(VkCommandBuffer commandBuffer, VkEvent event,
                                                     VkPipelineStageFlags2KHR stageMask, const ErrorObject& error_obj) const {
    return PreCallValidateCmdResetEvent2(commandBuffer, event, stageMask, error_obj);
}

void SyncValidator::PostCallRecordCmdResetEvent2KHR(VkCommandBuffer commandBuffer, VkEvent event,
                                                    VkPipelineStageFlags2KHR stageMask, const RecordObject& record_obj) {
    PostCallRecordCmdResetEvent2(commandBuffer, event, stageMask, record_obj);
}

void SyncValidator::PostCallRecordCmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags2 stageMask,
                                                 const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    auto event_state = Get<vvl::Event>(event);

    const SyncExecScope exec_scope = SyncExecScope::MakeSrc(queue_flags, stageMask);
    RecordCmdResetEvent(cb_context, std::move(event_state), exec_scope, record_obj.location);
}

bool SyncValidator::PreCallValidateCmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents,
                                                 VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
                                                 uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
                                                 uint32_t bufferMemoryBarrierCount,
                                                 const VkBufferMemoryBarrier* pBufferMemoryBarriers,
                                                 uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers,
                                                 const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    const SyncExecScope src_exec_scope = SyncExecScope::MakeSrc(queue_flags, srcStageMask);
    const SyncExecScope dst_exec_scope = SyncExecScope::MakeDst(queue_flags, dstStageMask);
    const BarrierSet barrier_set =
        BarrierSet(*this, src_exec_scope, dst_exec_scope, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount,
                   pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);

    std::vector<std::shared_ptr<const vvl::Event>> events(eventCount);
    for (uint32_t i = 0; i < eventCount; i++) {
        events[i] = Get<vvl::Event>(pEvents[i]);
    }

    EventCommand command;
    command.type = EventCommand::Type::kWait;
    command.events = events;
    command.barrier_sets.emplace_back(barrier_set);
    command.command = error_obj.location.function;
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::RecordCmdWaitEvents(CommandBufferContext& cb_context, std::vector<std::shared_ptr<const vvl::Event>>&& events,
                                        std::vector<BarrierSet>&& barrier_sets, const Location& loc) const {
    const ResourceUsageTag tag = cb_context.NextCommandTag(loc.function);
    EventCommand command;
    command.type = EventCommand::Type::kWait;
    command.events = events;
    command.barrier_sets = barrier_sets;
    command.command = loc.function;
    if (syncval_settings.IsRecordTimeValidationEnabled()) {
        command.Apply(cb_context.GetSyncEnvironment(), tag, cb_context.GetCurrentAccessContext());
    }
    cb_context.StoreCommand(tag, command);
    cb_context.AddReplayEntry(tag, false, WaitEventsReplay(std::move(events), std::move(barrier_sets), loc));
}

void SyncValidator::PostCallRecordCmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents,
                                                VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
                                                uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
                                                uint32_t bufferMemoryBarrierCount,
                                                const VkBufferMemoryBarrier* pBufferMemoryBarriers,
                                                uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers,
                                                const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    std::vector<std::shared_ptr<const vvl::Event>> events(eventCount);
    for (uint32_t i = 0; i < eventCount; i++) {
        events[i] = Get<vvl::Event>(pEvents[i]);
    }

    const SyncExecScope src_exec_scope = SyncExecScope::MakeSrc(queue_flags, srcStageMask);
    const SyncExecScope dst_exec_scope = SyncExecScope::MakeDst(queue_flags, dstStageMask);
    std::vector<BarrierSet> barrier_sets;
    barrier_sets.emplace_back(*this, src_exec_scope, dst_exec_scope, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount,
                              pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);

    RecordCmdWaitEvents(cb_context, std::move(events), std::move(barrier_sets), record_obj.location);
}

bool SyncValidator::PreCallValidateCmdWaitEvents2KHR(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents,
                                                     const VkDependencyInfoKHR* pDependencyInfos,
                                                     const ErrorObject& error_obj) const {
    return PreCallValidateCmdWaitEvents2(commandBuffer, eventCount, pEvents, pDependencyInfos, error_obj);
}

void SyncValidator::PostCallRecordCmdWaitEvents2KHR(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents,
                                                    const VkDependencyInfoKHR* pDependencyInfos, const RecordObject& record_obj) {
    PostCallRecordCmdWaitEvents2(commandBuffer, eventCount, pEvents, pDependencyInfos, record_obj);
}

bool SyncValidator::PreCallValidateCmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents,
                                                  const VkDependencyInfo* pDependencyInfos, const ErrorObject& error_obj) const {
    if (!pDependencyInfos || !syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    std::vector<std::shared_ptr<const vvl::Event>> events(eventCount);
    for (uint32_t i = 0; i < eventCount; i++) {
        events[i] = Get<vvl::Event>(pEvents[i]);
    }
    std::vector<BarrierSet> barrier_sets(eventCount);
    for (uint32_t i = 0; i < eventCount; i++) {
        barrier_sets[i] = BarrierSet(*this, queue_flags, pDependencyInfos[i]);
    }

    EventCommand command;
    command.type = EventCommand::Type::kWait;
    command.events = events;
    command.barrier_sets = barrier_sets;
    command.command = error_obj.location.function;
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents,
                                                 const VkDependencyInfo* pDependencyInfos, const RecordObject& record_obj) {
    if (!pDependencyInfos) {
        return;
    }
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const VkQueueFlags queue_flags = cb_state->GetQueueFlags();

    std::vector<std::shared_ptr<const vvl::Event>> events(eventCount);
    for (uint32_t i = 0; i < eventCount; i++) {
        events[i] = Get<vvl::Event>(pEvents[i]);
    }
    std::vector<BarrierSet> barrier_sets(eventCount);
    for (uint32_t i = 0; i < eventCount; i++) {
        barrier_sets[i] = BarrierSet(*this, queue_flags, pDependencyInfos[i]);
    }
    RecordCmdWaitEvents(cb_context, std::move(events), std::move(barrier_sets), record_obj.location);
}

bool SyncValidator::PreCallValidateCmdWriteBufferMarker2AMD(VkCommandBuffer commandBuffer, VkPipelineStageFlags2KHR pipelineStage,
                                                            VkBuffer dstBuffer, VkDeviceSize dstOffset, uint32_t marker,
                                                            const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto dst_buffer = Get<vvl::Buffer>(dstBuffer);
    if (!dst_buffer) {
        return false;
    }
    const BufferAccessCommand command{*dst_buffer, SYNC_COPY_TRANSFER_WRITE, MakeRange(dstOffset, 4), SyncFlag::kMarker,
                                      VK_NULL_HANDLE, "dstBuffer "};
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdWriteBufferMarker2AMD(VkCommandBuffer commandBuffer, VkPipelineStageFlags2KHR pipelineStage,
                                                           VkBuffer dstBuffer, VkDeviceSize dstOffset, uint32_t marker,
                                                           const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    auto dst_buffer = Get<vvl::Buffer>(dstBuffer);
    if (!dst_buffer) {
        return;
    }
    const ResourceUsageTagEx tag_ex = cb_context.AddCommandHandle(tag, dst_buffer->Handle());
    const BufferAccessCommand command{*dst_buffer, SYNC_COPY_TRANSFER_WRITE, MakeRange(dstOffset, 4), SyncFlag::kMarker,
                                      VK_NULL_HANDLE, "dstBuffer ", tag_ex.handle_index};
    if (syncval_settings.IsRecordTimeValidationEnabled()) {
        command.Apply(cb_context.GetSyncEnvironment(), tag, cb_context.GetCurrentAccessContext());
    }
    if (syncval_settings.full_validation) {
        cb_context.StoreCommand(tag, command);
    }
}

bool SyncValidator::PreCallValidateCmdExecuteCommands(VkCommandBuffer commandBuffer, uint32_t commandBufferCount,
                                                      const VkCommandBuffer* pCommandBuffers, const ErrorObject& error_obj) const {
    bool skip = false;
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);

    // Heavyweight, but we need a proxy copy of the active command buffer access context
    CommandBufferContext proxy_cb_context(cb_context, CommandBufferContext::AsProxyContext());

    auto& proxy_label_commands = proxy_cb_context.GetProxyLabelCommands();
    proxy_label_commands = cb_state->GetLabelCommands();

    // Make working copies of the access and events contexts
    for (uint32_t cb_index = 0; cb_index < commandBufferCount; ++cb_index) {
        if (cb_index == 0) {
            proxy_cb_context.NextCommandTag(error_obj.location.function, SubCommandType::kIndex);
        } else {
            proxy_cb_context.NextSubCommandTag(error_obj.location.function, SubCommandType::kIndex);
        }

        const auto recorded_cb = Get<vvl::CommandBuffer>(pCommandBuffers[cb_index]);
        if (!recorded_cb) {
            continue;
        }
        const CommandBufferContext& recorded_cb_context = GetCommandBufferContext(*recorded_cb);
        const ResourceUsageTag base_tag = proxy_cb_context.GetTagCount();
        const Location cb_loc = error_obj.location.dot(vvl::Field::pCommandBuffers, cb_index);

        // Update proxy label commands so they can be used by ImportRecordedAccessLog
        vvl::Append(proxy_label_commands, recorded_cb->GetLabelCommands());

        proxy_cb_context.ImportRecordedAccessLog(recorded_cb_context);

        if (syncval_settings.full_validation && recorded_cb_context.HasAllCommands()) {
            skip |= ReplayCommands(proxy_cb_context.GetSyncEnvironment(), proxy_cb_context.GetCbAccessContext(),
                                   recorded_cb_context, base_tag, cb_loc);
        } else {
            skip |= ValidateFirstUseHazards(proxy_cb_context.GetSyncEnvironment(), recorded_cb_context,
                                            proxy_cb_context.GetCbAccessContext(), base_tag, cb_loc);
            // The barriers have already been applied in ValidateFirstUseHazards
            proxy_cb_context.ResolveExecutedCommandBuffer(recorded_cb_context.GetCbAccessContext(), base_tag);
        }
    }
    proxy_label_commands.clear();
    return skip;
}

void SyncValidator::PostCallRecordBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset,
                                                  const RecordObject& record_obj) {
    if (record_obj.result != VK_SUCCESS) {
        return;
    }
    VkBindImageMemoryInfo bind_info = vku::InitStructHelper();
    bind_info.image = image;
    bind_info.memory = memory;
    bind_info.memoryOffset = memoryOffset;
    UpdateSyncImageMemoryBindState(1, &bind_info);
}

void SyncValidator::PostCallRecordBindImageMemory2(VkDevice device, uint32_t bindInfoCount, const VkBindImageMemoryInfo* pBindInfos,
                                                   const RecordObject& record_obj) {
    // Don't check |record_obj.result| as some binds might still be valid
    UpdateSyncImageMemoryBindState(bindInfoCount, pBindInfos);
}

void SyncValidator::PostCallRecordBindImageMemory2KHR(VkDevice device, uint32_t bindInfoCount,
                                                      const VkBindImageMemoryInfo* pBindInfos, const RecordObject& record_obj) {
    PostCallRecordBindImageMemory2(device, bindInfoCount, pBindInfos, record_obj);
}

void SyncValidator::PostCallRecordQueueWaitIdle(VkQueue queue, const RecordObject& record_obj) {
    if (record_obj.result != VK_SUCCESS || !syncval_settings.IsSubmitTimeProcessingEnabled() || queue == VK_NULL_HANDLE) {
        return;
    }
    const QueueId waited_queue = GetQueueId(queue);
    if (waited_queue == kQueueIdInvalid) {
        return;
    }
    const QueueState& queue_state = GetQueueState(waited_queue);
    ApplyTaggedWait(waited_queue, ResourceUsageRecord::kMaxIndex, queue_state.GetLastSynchronizedPresent(), {});

    // For each timeline, remove all signals signaled on the waited queue, except the last one.
    // The last signal is needed to represent the current timeline state.
    EnsureTimelineSignalsLimit(1, waited_queue);

    // Eliminate host waitable objects from the current queue.
    vvl::EraseIf(waitable_fences_, [waited_queue](const auto& sf) { return sf.second.queue_id == waited_queue; });
    for (auto& [semaphore, sync_points] : host_waitable_semaphores_) {
        vvl::EraseIf(sync_points, [waited_queue](const auto& sync_point) { return sync_point.queue_id == waited_queue; });
    }
}

void SyncValidator::PostCallRecordDeviceWaitIdle(VkDevice device, const RecordObject& record_obj) {
    const auto batches = GetAllQueueBatchContexts();

    // Collect information about last synchronized present over all queues
    LastSynchronizedPresent global_last_synchronized_present;
    for (const auto& batch : batches) {
        global_last_synchronized_present.Merge(batch->last_synchronized_present);
    }

    // Device wait will preserve unsynchronized present operations.
    for (const auto& batch : batches) {
        batch->ApplyDeviceWait(global_last_synchronized_present);
    }

    // For each timeline keep only the last signal per queue.
    // The last signal is needed to represent the current timeline state.
    EnsureTimelineSignalsLimit(1);

    // Cleanup fence waits associated with queues. Acquire fence waits are preserved
    auto is_queue_fence = [](const auto& waitable) { return waitable.second.queue_id != kQueueIdInvalid; };
    vvl::EraseIf(waitable_fences_, is_queue_fence);
    host_waitable_semaphores_.clear();
}

bool SyncValidator::PreCallValidateQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo,
                                                   const ErrorObject& error_obj) const {
    bool skip = false;
    if (!syncval_settings.IsSubmitTimeProcessingEnabled()) {
        return skip;
    }
    std::lock_guard lock_guard(queue_mutex_);
    skip |= const_cast<SyncValidator*>(this)->ProcessQueuePresent(queue, pPresentInfo, error_obj);
    return skip;
}

bool SyncValidator::ProcessQueuePresent(VkQueue queue, const VkPresentInfoKHR* pPresentInfo, const ErrorObject& error_obj) {
    bool skip = false;

    const QueueId queue_id = GetQueueId(queue);
    if (queue_id == kQueueIdInvalid) {
        return skip;
    }
    QueueState& queue_state = GetQueueState(queue_id);
    const uint64_t submit_id = queue_state.ReserveSubmitId();

    BatchContextPtr last_batch = queue_state.LastBatch();
    BatchContextPtr batch(std::make_shared<QueueBatchContext>(*this, queue_state));

    const auto wait_semaphores = vvl::make_span(pPresentInfo->pWaitSemaphores, pPresentInfo->waitSemaphoreCount);

    PresentedImages presented_images;
    uint32_t present_tag_count = SetupPresentInfo(*pPresentInfo, batch, presented_images);

    SignalsUpdate signals_update(*this);
    auto resolved_batches = batch->ResolvePresentWaits(wait_semaphores, presented_images, signals_update);

    // Import the previous batch information
    if (last_batch && !vvl::Contains(resolved_batches, last_batch)) {
        batch->ResolveLastBatch(last_batch);
        resolved_batches.emplace_back(std::move(last_batch));
    }

    // The purpose of keeping return value is to ensure async batches are alive during validation.
    // Validation accesses raw pointer to async contexts stored in AsyncReference.
    const auto async_batches = batch->RegisterAsyncContexts(resolved_batches);

    // Convert present tags to global range
    const ResourceUsageTag global_range_start = batch->SetupBatchTags(present_tag_count);
    for (PresentedImage& presented : presented_images) {
        presented.tag += global_range_start;
    }

    skip |= batch->DoQueuePresentValidate(error_obj.location, presented_images);
    batch->DoPresentOperations(presented_images);
    batch->LogPresentOperations(presented_images, submit_id);

    // Update state if there are no validation errors
    if (!skip) {
        stats.UpdateAccessStats(*this);
        stats.UpdateMemoryStats();
        queue_state.SetLastBatch(std::move(batch));
        ApplySignalsUpdate(signals_update, queue_state.LastBatch());
        for (auto& presented : presented_images) {
            presented.ExportToSwapchain();
        }
    }
    return skip;
}

uint32_t SyncValidator::SetupPresentInfo(const VkPresentInfoKHR& present_info, BatchContextPtr& batch,
                                         PresentedImages& presented_images) {
    const VkSwapchainKHR* swapchains = present_info.pSwapchains;
    const uint32_t* image_indices = present_info.pImageIndices;
    const uint32_t swapchain_count = present_info.swapchainCount;

    presented_images.reserve(swapchain_count);
    for (uint32_t present_index = 0; present_index < swapchain_count; present_index++) {
        // Note: Given the "EraseIf" implementation for acquire fence waits, each presentation needs a unique tag.
        const ResourceUsageTag tag = presented_images.size();
        presented_images.emplace_back(*this, batch, swapchains[present_index], image_indices[present_index], present_index, tag);
        if (presented_images.back().Invalid()) {
            presented_images.pop_back();
        }
    }
    // Present is tagged for each swapchain.
    return static_cast<uint32_t>(presented_images.size());
}

void SyncValidator::PostCallRecordAcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout,
                                                      VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex,
                                                      const RecordObject& record_obj) {
    if (!syncval_settings.IsSubmitTimeProcessingEnabled()) {
        return;
    }
    RecordAcquireNextImageState(device, swapchain, timeout, semaphore, fence, pImageIndex, record_obj);
}

void SyncValidator::PostCallRecordAcquireNextImage2KHR(VkDevice device, const VkAcquireNextImageInfoKHR* pAcquireInfo,
                                                       uint32_t* pImageIndex, const RecordObject& record_obj) {
    if (!syncval_settings.IsSubmitTimeProcessingEnabled()) {
        return;
    }
    RecordAcquireNextImageState(device, pAcquireInfo->swapchain, pAcquireInfo->timeout, pAcquireInfo->semaphore,
                                pAcquireInfo->fence, pImageIndex, record_obj);
}

void SyncValidator::RecordAcquireNextImageState(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore,
                                                VkFence fence, uint32_t* pImageIndex, const RecordObject& record_obj) {
    if ((VK_SUCCESS != record_obj.result) && (VK_SUBOPTIMAL_KHR != record_obj.result)) {
        return;
    }

    // Get the image out of the presented list and create apppropriate fences/semaphores.
    auto swapchain_base = Get<vvl::Swapchain>(swapchain);
    if (vvl::StateObject::Invalid(swapchain_base)) {
        return;
    }  // Invalid acquire calls to be caught in CoreCheck/Parameter validation

    auto& swapchain_state = SubState(*swapchain_base);

    PresentedImage presented = swapchain_state.MovePresentedImage(*pImageIndex);
    if (presented.Invalid()) {
        return;
    }

    // No way to make access safe, so nothing to record
    if ((semaphore == VK_NULL_HANDLE) && (fence == VK_NULL_HANDLE)) {
        return;
    }

    // We create a queue-less QBC for the Semaphore and fences to wait on

    // Note: this is a heavyweight way to deal with the fact that all operation logs live in the QueueBatchContext... and
    // acquire doesn't happen on a queue, but we need a place to put the acquire operation access record.
    auto batch = std::make_shared<QueueBatchContext>(*this);
    batch->SetupAccessContext(presented);
    const ResourceUsageTag acquire_tag = batch->SetupBatchTags(1);
    batch->DoAcquireOperation(presented);
    batch->LogAcquireOperation(presented, record_obj.location.function);

    // Now swap out the present queue batch with the acquired one.
    // Note that fence and signal will read the acquire batch from presented, so this needs to be done before
    // setting up the synchronization
    presented.batch = std::move(batch);

    if (semaphore != VK_NULL_HANDLE) {
        std::shared_ptr<const vvl::Semaphore> sem_state = Get<vvl::Semaphore>(semaphore);
        if (sem_state) {
            // This will ignore any duplicated signal (emplace does not update existing entry),
            // and the core validation reports and error in this case.
            binary_signals_.emplace(sem_state->VkHandle(), SignalInfo(sem_state, presented, acquire_tag));
        }
    }
    if (fence != VK_NULL_HANDLE) {
        FenceHostSyncPoint sync_point;
        sync_point.tag = acquire_tag;
        sync_point.acquired = AcquiredImage(presented, acquire_tag);
        UpdateFenceHostSyncPoint(fence, std::move(sync_point));
    }
}

bool SyncValidator::PreCallValidateQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence,
                                               const ErrorObject& error_obj) const {
    bool skip = false;
    if (!syncval_settings.IsSubmitTimeProcessingEnabled()) {
        return skip;
    }

    SubmitInfoArrayConverter submit_info(pSubmits, submitCount);
    const VkSubmitInfo2* submits = submit_info.submit_infos2.data();

    std::lock_guard lock_guard(queue_mutex_);
    skip |= const_cast<SyncValidator*>(this)->ProcessQueueSubmit(queue, submitCount, submits, fence, error_obj);
    return skip;
}

bool SyncValidator::PreCallValidateQueueSubmit2(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2* pSubmits, VkFence fence,
                                                const ErrorObject& error_obj) const {
    bool skip = false;
    if (!syncval_settings.IsSubmitTimeProcessingEnabled()) {
        return skip;
    }
    std::lock_guard lock_guard(queue_mutex_);
    skip |= const_cast<SyncValidator*>(this)->ProcessQueueSubmit(queue, submitCount, pSubmits, fence, error_obj);
    return skip;
}

bool SyncValidator::PreCallValidateQueueSubmit2KHR(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2KHR* pSubmits,
                                                   VkFence fence, const ErrorObject& error_obj) const {
    return PreCallValidateQueueSubmit2(queue, submitCount, pSubmits, fence, error_obj);
}

static std::vector<CommandBufferConstPtr> GetCommandBuffers(const vvl::DeviceState& device_state,
                                                            const VkSubmitInfo2& submit_info) {
    // Collected command buffers have the same indexing as in the input VkSubmitInfo2 for reporting purposes.
    // If Get query returns null, it is stored in the result array to keep original indexing.
    std::vector<CommandBufferConstPtr> command_buffers;
    command_buffers.reserve(submit_info.commandBufferInfoCount);
    for (const auto& cb_info : vvl::make_span(submit_info.pCommandBufferInfos, submit_info.commandBufferInfoCount)) {
        command_buffers.emplace_back(device_state.Get<vvl::CommandBuffer>(cb_info.commandBuffer));
    }
    return command_buffers;
}

bool SyncValidator::ProcessQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2* pSubmits, VkFence fence,
                                       const ErrorObject& error_obj) {
    bool skip = false;

    const QueueId queue_id = GetQueueId(queue);
    if (queue_id == kQueueIdInvalid) {
        return skip;
    }

    QueueState& queue_state = GetQueueState(queue_id);
    SignalsUpdate signals_update(*this);

    // The submit id is a mutable automic which is not recoverable on a skip == true condition
    uint64_t submit_id = queue_state.ReserveSubmitId();

    // Update label stack as we progress through batches and command buffers
    auto current_label_stack = queue_state.GetQueue()->cmdbuf_label_stack;

    BatchContextPtr last_batch = queue_state.LastBatch();
    bool has_unresolved_batches = !queue_state.UnresolvedBatches().empty();

    BatchContextPtr new_last_batch;
    std::vector<UnresolvedBatch> new_unresolved_batches;
    bool new_timeline_signals = false;

    for (uint32_t batch_idx = 0; batch_idx < submitCount; batch_idx++) {
        const VkSubmitInfo2& submit = pSubmits[batch_idx];
        auto batch = std::make_shared<QueueBatchContext>(*this, queue_state);

        const auto wait_semaphores = vvl::make_span(submit.pWaitSemaphoreInfos, submit.waitSemaphoreInfoCount);
        std::vector<VkSemaphoreSubmitInfo> unresolved_waits;
        auto resolved_batches = batch->ResolveSubmitWaits(wait_semaphores, unresolved_waits, signals_update);

        // Add unresolved batch
        if (has_unresolved_batches || !unresolved_waits.empty()) {
            UnresolvedBatch unresolved_batch;
            unresolved_batch.batch = std::move(batch);
            unresolved_batch.submit_func = error_obj.location.function;
            unresolved_batch.submit_index = submit_id;
            unresolved_batch.batch_index = batch_idx;
            unresolved_batch.command_buffers = GetCommandBuffers(*device_state, submit);
            unresolved_batch.unresolved_waits = std::move(unresolved_waits);
            unresolved_batch.resolved_dependencies = std::move(resolved_batches);
            if (submit.pSignalSemaphoreInfos && submit.signalSemaphoreInfoCount) {
                const auto last_info = submit.pSignalSemaphoreInfos + submit.signalSemaphoreInfoCount;
                unresolved_batch.signals.assign(submit.pSignalSemaphoreInfos, last_info);
            }
            unresolved_batch.label_stack = current_label_stack;
            new_unresolved_batches.emplace_back(std::move(unresolved_batch));
            has_unresolved_batches = true;
            stats.AddUnresolvedBatch();
            continue;
        }
        new_last_batch = batch;

        // Import the previous batch information
        if (last_batch && !vvl::Contains(resolved_batches, last_batch)) {
            batch->ResolveLastBatch(last_batch);
            resolved_batches.emplace_back(std::move(last_batch));
        }

        // The purpose of keeping return value is to ensure async batches are alive during validation.
        // Validation accesses raw pointer to async contexts stored in AsyncReference.
        // TODO: All syncval tests pass when the return value is ignored. Write a regression test that fails/crashes in this case.
        const auto async_batches = batch->RegisterAsyncContexts(resolved_batches);

        const auto command_buffers = GetCommandBuffers(*device_state, submit);
        skip |= batch->ValidateSubmit(command_buffers, submit_id, batch_idx, current_label_stack, error_obj.location);

        const auto submit_signals = vvl::make_span(submit.pSignalSemaphoreInfos, submit.signalSemaphoreInfoCount);
        new_timeline_signals |= signals_update.RegisterSignals(batch, submit_signals);

        // Unless the previous batch was referenced by a signal it will self destruct
        // in the record phase when the last batch is updated.
        last_batch = batch;
    }

    if (new_last_batch && !skip) {
        queue_state.SetLastBatch(std::move(new_last_batch));
    }
    if (!new_unresolved_batches.empty() && !skip) {
        vvl::Append(queue_state.UnresolvedBatches(), new_unresolved_batches);
    }

    // Check if timeline signals resolve existing wait-before-signal dependencies
    if (new_timeline_signals) {
        skip |= PropagateTimelineSignals(signals_update);
    }

    if (!skip) {
        stats.UpdateMemoryStats();
        ApplySignalsUpdate(signals_update, queue_state.LastBatch());
        FenceHostSyncPoint sync_point;
        sync_point.queue_id = queue_state.GetQueueId();
        sync_point.tag = ReserveGlobalTagRange(1).begin;
        if (queue_state.LastBatch()) {
            sync_point.queue_sync_tags = queue_state.LastBatch()->GetQueueSyncTags();
        }
        UpdateFenceHostSyncPoint(fence, std::move(sync_point));
    }
    return skip;
}

bool SyncValidator::PropagateTimelineSignals(SignalsUpdate& signals_update) {
    bool skip = false;

    // The caller ensures we just registered new timeline signals
    bool new_timeline_signals = true;

    // Each iteration uses registered timeline signals to resolve batches.
    // If a resolved batch generates new timeline signals, the loop runs again
    while (new_timeline_signals) {
        new_timeline_signals = false;

        for (QueueState& queue_state : queue_states_) {
            auto& unresolved_batches = queue_state.UnresolvedBatches();
            if (unresolved_batches.empty()) {
                continue;
            }
            // Resolve waits that have matching signal
            for (UnresolvedBatch& unresolved_batch : unresolved_batches) {
                auto it = unresolved_batch.unresolved_waits.begin();
                while (it != unresolved_batch.unresolved_waits.end()) {
                    const VkSemaphoreSubmitInfo& wait_info = *it;
                    auto resolving_signal = signals_update.OnTimelineWait(wait_info.semaphore, wait_info.value);
                    if (!resolving_signal.has_value()) {
                        ++it;
                        continue;  // resolving signal not found, the wait stays unresolved
                    }
                    if (resolving_signal->batch) {  // null for host signals
                        unresolved_batch.batch->ResolveSubmitSemaphoreWait(*resolving_signal, wait_info.stageMask);
                        unresolved_batch.batch->ImportTags(*resolving_signal->batch);
                        unresolved_batch.resolved_dependencies.emplace_back(resolving_signal->batch);
                    }
                    it = unresolved_batch.unresolved_waits.erase(it);
                }
            }

            BatchContextPtr last_batch = queue_state.LastBatch();
            const BatchContextPtr initial_last_batch = last_batch;

            // Process batches that do not have unresolved waits anymore.
            // Stop when find a batch with unresolved waits or when all the batches are processed.
            while (!unresolved_batches.empty() && unresolved_batches.front().unresolved_waits.empty()) {
                UnresolvedBatch& ready_batch = unresolved_batches.front();

                // Import the previous batch information
                if (last_batch && !vvl::Contains(ready_batch.resolved_dependencies, last_batch)) {
                    ready_batch.batch->ResolveLastBatch(last_batch);
                    ready_batch.resolved_dependencies.emplace_back(std::move(last_batch));
                }

                const auto async_batches = ready_batch.batch->RegisterAsyncContexts(ready_batch.resolved_dependencies);

                skip |= ready_batch.batch->ValidateSubmit(ready_batch.command_buffers, ready_batch.submit_index,
                                                          ready_batch.batch_index, ready_batch.label_stack,
                                                          Location(ready_batch.submit_func));

                // Process signals. New timeline signals can resolve more batches on the next iteration
                const auto submit_signals = vvl::make_span(ready_batch.signals.data(), ready_batch.signals.size());
                new_timeline_signals |= signals_update.RegisterSignals(ready_batch.batch, submit_signals);

                last_batch = ready_batch.batch;
                unresolved_batches.erase(unresolved_batches.begin());
                stats.RemoveUnresolvedBatch();
            }

            if (last_batch != initial_last_batch) {
                queue_state.SetLastBatch(std::move(last_batch));
            }
        }
    }
    return skip;
}

void SyncValidator::PostCallRecordGetFenceStatus(VkDevice device, VkFence fence, const RecordObject& record_obj) {
    if (!syncval_settings.IsSubmitTimeProcessingEnabled()) {
        return;
    }
    if (record_obj.result == VK_SUCCESS) {
        // fence is signalled, mark it as waited for
        WaitForFence(fence);
    }
}

void SyncValidator::PostCallRecordWaitForFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll,
                                                uint64_t timeout, const RecordObject& record_obj) {
    if (!syncval_settings.IsSubmitTimeProcessingEnabled()) {
        return;
    }
    if ((record_obj.result == VK_SUCCESS) && ((VK_TRUE == waitAll) || (1 == fenceCount))) {
        // We can only know the pFences have signal if we waited for all of them, or there was only one of them
        for (uint32_t i = 0; i < fenceCount; i++) {
            WaitForFence(pFences[i]);
        }
    }
}

bool SyncValidator::PreCallValidateSignalSemaphore(VkDevice device, const VkSemaphoreSignalInfo* pSignalInfo,
                                                   const ErrorObject& error_obj) const {
    bool skip = false;
    if (!syncval_settings.IsSubmitTimeProcessingEnabled()) {
        return skip;
    }
    // Although SignalSemaphore does not run on the queue, the signalling can resolve
    // previously submitted batches that are waiting for this signal, and this will
    // initiate validation that touches queue state. That's why queue mutex is used here.
    std::lock_guard lock_guard(queue_mutex_);
    skip |= const_cast<SyncValidator*>(this)->ProcessSignalSemaphore(device, pSignalInfo);
    return skip;
}

bool SyncValidator::PreCallValidateSignalSemaphoreKHR(VkDevice device, const VkSemaphoreSignalInfo* pSignalInfo,
                                                      const ErrorObject& error_obj) const {
    return PreCallValidateSignalSemaphore(device, pSignalInfo, error_obj);
}

bool SyncValidator::ProcessSignalSemaphore(VkDevice device, const VkSemaphoreSignalInfo* pSignalInfo) {
    bool skip = false;

    auto semaphore_state = Get<vvl::Semaphore>(pSignalInfo->semaphore);
    if (!semaphore_state) {
        return skip;
    }

    SignalsUpdate signals_update(*this);
    std::vector<SignalInfo>& signals = signals_update.timeline_signals[pSignalInfo->semaphore];

    // Reject invalid signal
    if (!signals.empty() && pSignalInfo->value <= signals.back().timeline_value) {
        return skip;  // [core validation check]: strictly increasing signal values
    }

    signals.emplace_back(SignalInfo(semaphore_state, pSignalInfo->value));
    skip |= PropagateTimelineSignals(signals_update);

    if (!skip) {
        ApplySignalsUpdate(signals_update, nullptr);
    }
    return skip;
}

void SyncValidator::PostCallRecordWaitSemaphores(VkDevice device, const VkSemaphoreWaitInfo* pWaitInfo, uint64_t timeout,
                                                 const RecordObject& record_obj) {
    if (!syncval_settings.IsSubmitTimeProcessingEnabled()) {
        return;
    }
    const bool wait_all = pWaitInfo->semaphoreCount == 1 || (pWaitInfo->flags & VK_SEMAPHORE_WAIT_ANY_BIT) == 0;
    if (record_obj.result == VK_SUCCESS && wait_all) {
        for (uint32_t i = 0; i < pWaitInfo->semaphoreCount; i++) {
            WaitForSemaphore(pWaitInfo->pSemaphores[i], pWaitInfo->pValues[i]);
        }
    }
}

void SyncValidator::PostCallRecordWaitSemaphoresKHR(VkDevice device, const VkSemaphoreWaitInfo* pWaitInfo, uint64_t timeout,
                                                    const RecordObject& record_obj) {
    PostCallRecordWaitSemaphores(device, pWaitInfo, timeout, record_obj);
}

void SyncValidator::PostCallRecordGetSemaphoreCounterValue(VkDevice device, VkSemaphore semaphore, uint64_t* pValue,
                                                           const RecordObject& record_obj) {
    if (!syncval_settings.IsSubmitTimeProcessingEnabled()) {
        return;
    }
    if (record_obj.result == VK_SUCCESS) {
        WaitForSemaphore(semaphore, *pValue);
    }
}

void SyncValidator::PostCallRecordGetSemaphoreCounterValueKHR(VkDevice device, VkSemaphore semaphore, uint64_t* pValue,
                                                              const RecordObject& record_obj) {
    PostCallRecordGetSemaphoreCounterValue(device, semaphore, pValue, record_obj);
}

// Returns null when device address is asssociated with no buffers or more than one buffer.
// Otherwise returns a valid buffer (device address is associated with a single buffer).
// When syncval adds memory aliasing support the need of this function can be revisited.
static const vvl::Buffer* GetSingleBufferFromDeviceAddress(const vvl::DeviceState& device, VkDeviceAddress device_address) {
    vvl::span<vvl::Buffer* const> buffers = device.GetBuffersByAddress(device_address);
    if (buffers.empty()) {
        return nullptr;
    }
    if (buffers.size() > 1) {  // memory aliasing use case
        return nullptr;
    }
    return buffers[0];
}

struct AccelerationStructureGeometryInfo {
    const vvl::Buffer* vertex_data = nullptr;
    AccessRange vertex_range;
    const vvl::Buffer* index_data = nullptr;
    AccessRange index_range;
    const vvl::Buffer* transform_data = nullptr;
    AccessRange transform_range;
    const vvl::Buffer* aabb_data = nullptr;
    AccessRange aabb_range;
    const vvl::Buffer* instance_data = nullptr;
    AccessRange instance_range;
};

static std::optional<AccelerationStructureGeometryInfo> GetValidGeometryInfo(
    const vvl::DeviceState& device, const VkAccelerationStructureGeometryKHR& geometry,
    const VkAccelerationStructureBuildRangeInfoKHR& range_info) {
    if (geometry.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
        const VkAccelerationStructureGeometryTrianglesDataKHR& triangles = geometry.geometry.triangles;
        AccelerationStructureGeometryInfo geometry_info;

        // Assume that synchronization ranges cover the entire vertex struct,
        // even if positional data is strided (i.e., interleaved with other attributes).
        // That is, the application does not attempt to synchronize each position variable
        // with a separate buffer barrier range.
        const vvl::Buffer* p_vertex_data = GetSingleBufferFromDeviceAddress(device, triangles.vertexData.deviceAddress);

        if (triangles.indexType == VK_INDEX_TYPE_NONE_KHR) {
            // Vertex data
            if (p_vertex_data) {
                geometry_info.vertex_data = p_vertex_data;
                const VkDeviceSize base_vertex_offset = triangles.vertexData.deviceAddress - p_vertex_data->deviceAddress;
                const VkDeviceSize local_offset = range_info.primitiveOffset + range_info.firstVertex * triangles.vertexStride;
                const VkDeviceSize offset = base_vertex_offset + local_offset;
                const VkDeviceSize vertex_data_size = 3 * range_info.primitiveCount * triangles.vertexStride;
                geometry_info.vertex_range = MakeRange(*p_vertex_data, offset, vertex_data_size);
            }
        } else {
            // Vertex data
            if (p_vertex_data) {
                geometry_info.vertex_data = p_vertex_data;
                const VkDeviceSize base_vertex_offset = triangles.vertexData.deviceAddress - p_vertex_data->deviceAddress;
                const VkDeviceSize local_offset = range_info.firstVertex * triangles.vertexStride;
                const VkDeviceSize offset = base_vertex_offset + local_offset;
                const VkDeviceSize all_vertex_data_size = (triangles.maxVertex + 1) * triangles.vertexStride;
                const VkDeviceSize potentially_accessed_vertex_data_size = all_vertex_data_size - local_offset;
                geometry_info.vertex_range = MakeRange(*p_vertex_data, offset, potentially_accessed_vertex_data_size);
            }
            // Index data
            const auto p_index_data = GetSingleBufferFromDeviceAddress(device, triangles.indexData.deviceAddress);
            if (p_index_data) {
                geometry_info.index_data = p_index_data;
                const VkDeviceSize base_index_offset = triangles.indexData.deviceAddress - p_index_data->deviceAddress;
                const uint32_t index_byte_size = IndexTypeByteSize(triangles.indexType);
                const VkDeviceSize offset = base_index_offset + range_info.primitiveOffset;
                const uint32_t index_data_size = 3 * range_info.primitiveCount * index_byte_size;
                geometry_info.index_range = MakeRange(*p_index_data, offset, index_data_size);
            }
        }
        // Transform data
        if (const vvl::Buffer* p_transform_data = GetSingleBufferFromDeviceAddress(device, triangles.transformData.deviceAddress)) {
            const VkDeviceSize base_offset = triangles.transformData.deviceAddress - p_transform_data->deviceAddress;
            const VkDeviceSize offset = base_offset + range_info.transformOffset;
            geometry_info.transform_data = p_transform_data;
            geometry_info.transform_range = MakeRange(*p_transform_data, offset, sizeof(VkTransformMatrixKHR));
        }
        return geometry_info;
    } else if (geometry.geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
        // Make a similar assumption for strided aabb data as for vertex data - synchronization ranges themselves are not strided.
        const VkAccelerationStructureGeometryAabbsDataKHR& aabbs = geometry.geometry.aabbs;
        if (const vvl::Buffer* p_aabbs = GetSingleBufferFromDeviceAddress(device, aabbs.data.deviceAddress)) {
            AccelerationStructureGeometryInfo geometry_info;
            geometry_info.aabb_data = p_aabbs;
            const VkDeviceSize base_offset = aabbs.data.deviceAddress - p_aabbs->deviceAddress;
            const VkDeviceSize offset = base_offset + range_info.primitiveOffset;
            const VkDeviceSize aabb_data_size = range_info.primitiveCount * sizeof(VkAabbPositionsKHR);
            geometry_info.aabb_range = MakeRange(*p_aabbs, offset, aabb_data_size);
            return geometry_info;
        }
    } else if (geometry.geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
        const VkAccelerationStructureGeometryInstancesDataKHR& instances = geometry.geometry.instances;
        if (const vvl::Buffer* p_instances = GetSingleBufferFromDeviceAddress(device, instances.data.deviceAddress)) {
            AccelerationStructureGeometryInfo geometry_info;
            geometry_info.instance_data = p_instances;
            const VkDeviceSize base_offset = instances.data.deviceAddress - p_instances->deviceAddress;
            const VkDeviceSize offset = base_offset + range_info.primitiveOffset;
            const VkDeviceSize instance_data_size =
                range_info.primitiveCount *
                (instances.arrayOfPointers ? sizeof(VkDeviceAddress) : sizeof(VkAccelerationStructureInstanceKHR));
            geometry_info.instance_range = MakeRange(*p_instances, offset, instance_data_size);
            return geometry_info;
        }
    }
    return {};
}

ResourceAccessCommand SyncValidator::MakeAccelerationStructureAccessCommand(VkAccelerationStructureKHR acceleration_structure,
                                                                            SyncAccessIndex access_index,
                                                                            const Location& loc) const {
    ResourceAccessCommand command;
    const auto acceleration_structure_state = Get<vvl::AccelerationStructureKHR>(acceleration_structure);
    if (!acceleration_structure_state) {
        return command;
    }
    const vvl::BufferAndOffset buffer = acceleration_structure_state->GetFirstValidBuffer(*device_state);
    if (!buffer) {
        return command;
    }

    ResourceAccessCommand::BufferAccess access;
    access.buffer = buffer.state;
    access.access_index = access_index;
    access.range = MakeRange(buffer.offset, acceleration_structure_state->GetSize());
    access.tag_handle = buffer.state->Handle();
    access.resource_name.clear();
    access.acceleration_structure_info.emplace(ResourceAccessCommand::BufferAccess::AccelerationStructureInfo{
        acceleration_structure_state.get(), loc.Fields()});
    command.accesses.emplace_back(std::move(access));
    return command;
}

ResourceAccessCommand SyncValidator::MakeBuildAccelerationStructuresAccessCommand(
    uint32_t info_count, const VkAccelerationStructureBuildGeometryInfoKHR* infos,
    const VkAccelerationStructureBuildRangeInfoKHR* const* build_range_infos, const Location& loc) const {
    ResourceAccessCommand command;
    auto add_buffer = [&](const vvl::Buffer& buffer, SyncAccessIndex access_index, const AccessRange& range,
                          const char* resource_name) -> ResourceAccessCommand::BufferAccess& {
        ResourceAccessCommand::BufferAccess access;
        access.buffer = &buffer;
        access.access_index = access_index;
        access.range = range;
        access.tag_handle = buffer.Handle();
        access.resource_name = std::string(resource_name) + " ";
        command.accesses.emplace_back(std::move(access));
        return std::get<ResourceAccessCommand::BufferAccess>(command.accesses.back());
    };

    for (const auto [i, info] : vvl::enumerate(infos, info_count)) {
        if (const vvl::Buffer* scratch_buffer = GetSingleBufferFromDeviceAddress(*device_state, info.scratchData.deviceAddress)) {
            const VkDeviceSize scratch_size = rt::ComputeScratchSize(rt::BuildType::Device, device, info, build_range_infos[i]);
            const VkDeviceSize offset = info.scratchData.deviceAddress - scratch_buffer->deviceAddress;
            add_buffer(*scratch_buffer, SYNC_ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_WRITE,
                       MakeRange(*scratch_buffer, offset, scratch_size), "scratch buffer");
        }

        const auto src_acceleration_structure = Get<vvl::AccelerationStructureKHR>(info.srcAccelerationStructure);
        const auto dst_acceleration_structure = Get<vvl::AccelerationStructureKHR>(info.dstAccelerationStructure);
        auto src_command = MakeAccelerationStructureAccessCommand(info.srcAccelerationStructure,
                                                                  SYNC_ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_READ,
                                                                  loc.dot(Field::pInfos, i).dot(Field::srcAccelerationStructure));
        if (src_acceleration_structure && src_acceleration_structure == dst_acceleration_structure &&
            !src_command.accesses.empty()) {
            std::get<ResourceAccessCommand::BufferAccess>(src_command.accesses.front()).apply_access = false;
        }
        command.Append(std::move(src_command));
        command.Append(MakeAccelerationStructureAccessCommand(info.dstAccelerationStructure,
                                                              SYNC_ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_WRITE,
                                                              loc.dot(Field::pInfos, i).dot(Field::dstAccelerationStructure)));

        const VkAccelerationStructureBuildRangeInfoKHR* range_infos = build_range_infos[i];
        if (!range_infos) {
            continue;
        }
        for (uint32_t geometry_index = 0; geometry_index < info.geometryCount; ++geometry_index) {
            const auto* geometry = info.pGeometries ? &info.pGeometries[geometry_index] : info.ppGeometries[geometry_index];
            if (!geometry) {
                continue;
            }
            const auto geometry_info = GetValidGeometryInfo(*device_state, *geometry, range_infos[geometry_index]);
            if (!geometry_info) {
                continue;
            }
            if (geometry_info->vertex_data) {
                add_buffer(*geometry_info->vertex_data, SYNC_ACCELERATION_STRUCTURE_BUILD_SHADER_READ,
                           geometry_info->vertex_range, "vertex data");
            }
            if (geometry_info->index_data) {
                add_buffer(*geometry_info->index_data, SYNC_ACCELERATION_STRUCTURE_BUILD_SHADER_READ,
                           geometry_info->index_range, "index data");
            }
            if (geometry_info->transform_data) {
                add_buffer(*geometry_info->transform_data, SYNC_ACCELERATION_STRUCTURE_BUILD_SHADER_READ,
                           geometry_info->transform_range, "transform data");
            }
            if (geometry_info->aabb_data) {
                add_buffer(*geometry_info->aabb_data, SYNC_ACCELERATION_STRUCTURE_BUILD_SHADER_READ,
                           geometry_info->aabb_range, "aabb data");
            }
            if (geometry_info->instance_data) {
                add_buffer(*geometry_info->instance_data, SYNC_ACCELERATION_STRUCTURE_BUILD_SHADER_READ,
                           geometry_info->instance_range, "instance data");
            }
        }
    }
    return command;
}

bool SyncValidator::PreCallValidateCmdBuildAccelerationStructuresKHR(
    VkCommandBuffer commandBuffer, uint32_t infoCount, const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
    const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos, const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const auto command = MakeBuildAccelerationStructuresAccessCommand(infoCount, pInfos, ppBuildRangeInfos, error_obj.location);
    return command.Validate(cb_context, error_obj.location);
}
void SyncValidator::PostCallRecordCmdBuildAccelerationStructuresKHR(
    VkCommandBuffer commandBuffer, uint32_t infoCount, const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
    const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    auto command = MakeBuildAccelerationStructuresAccessCommand(infoCount, pInfos, ppBuildRangeInfos, record_obj.location);
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}
bool SyncValidator::PreCallValidateCmdCopyAccelerationStructureKHR(VkCommandBuffer commandBuffer,
                                                                   const VkCopyAccelerationStructureInfoKHR* pInfo,
                                                                   const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto command = MakeAccelerationStructureAccessCommand(pInfo->src, SYNC_ACCELERATION_STRUCTURE_COPY_ACCELERATION_STRUCTURE_READ,
                                                          error_obj.location.dot(Field::pInfo).dot(Field::src));
    command.Append(MakeAccelerationStructureAccessCommand(pInfo->dst, SYNC_ACCELERATION_STRUCTURE_COPY_ACCELERATION_STRUCTURE_WRITE,
                                                          error_obj.location.dot(Field::pInfo).dot(Field::dst)));
    return command.Validate(cb_context, error_obj.location);
}
void SyncValidator::PostCallRecordCmdCopyAccelerationStructureKHR(VkCommandBuffer commandBuffer,
                                                                  const VkCopyAccelerationStructureInfoKHR* pInfo,
                                                                  const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    auto command = MakeAccelerationStructureAccessCommand(pInfo->src, SYNC_ACCELERATION_STRUCTURE_COPY_ACCELERATION_STRUCTURE_READ,
                                                          record_obj.location.dot(Field::pInfo).dot(Field::src));
    command.Append(MakeAccelerationStructureAccessCommand(pInfo->dst, SYNC_ACCELERATION_STRUCTURE_COPY_ACCELERATION_STRUCTURE_WRITE,
                                                          record_obj.location.dot(Field::pInfo).dot(Field::dst)));
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}
bool SyncValidator::PreCallValidateCmdCopyAccelerationStructureToMemoryKHR(
    VkCommandBuffer commandBuffer, const VkCopyAccelerationStructureToMemoryInfoKHR* pInfo,
    const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const auto command =
        MakeAccelerationStructureAccessCommand(pInfo->src, SYNC_ACCELERATION_STRUCTURE_COPY_ACCELERATION_STRUCTURE_READ,
                                               error_obj.location.dot(Field::pInfo).dot(Field::src));
    return command.Validate(cb_context, error_obj.location);
}
void SyncValidator::PostCallRecordCmdCopyAccelerationStructureToMemoryKHR(
    VkCommandBuffer commandBuffer, const VkCopyAccelerationStructureToMemoryInfoKHR* pInfo,
    const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    auto command = MakeAccelerationStructureAccessCommand(pInfo->src, SYNC_ACCELERATION_STRUCTURE_COPY_ACCELERATION_STRUCTURE_READ,
                                                          record_obj.location.dot(Field::pInfo).dot(Field::src));
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}
bool SyncValidator::PreCallValidateCmdCopyMemoryToAccelerationStructureKHR(
    VkCommandBuffer commandBuffer, const VkCopyMemoryToAccelerationStructureInfoKHR* pInfo,
    const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    const auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const auto command =
        MakeAccelerationStructureAccessCommand(pInfo->dst, SYNC_ACCELERATION_STRUCTURE_COPY_ACCELERATION_STRUCTURE_WRITE,
                                               error_obj.location.dot(Field::pInfo).dot(Field::dst));
    return command.Validate(cb_context, error_obj.location);
}
void SyncValidator::PostCallRecordCmdCopyMemoryToAccelerationStructureKHR(
    VkCommandBuffer commandBuffer, const VkCopyMemoryToAccelerationStructureInfoKHR* pInfo,
    const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    auto command = MakeAccelerationStructureAccessCommand(pInfo->dst, SYNC_ACCELERATION_STRUCTURE_COPY_ACCELERATION_STRUCTURE_WRITE,
                                                          record_obj.location.dot(Field::pInfo).dot(Field::dst));
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}
ResourceAccessCommand SyncValidator::MakeSbtBufferAccessCommand(const VkStridedDeviceAddressRegionKHR* sbt_address_region,
                                                                const char* sbt_buffer_label) const {
    ResourceAccessCommand command;
    if (!sbt_address_region) {
        return command;
    }
    const vvl::Buffer* buffer = GetSingleBufferFromDeviceAddress(*device_state, sbt_address_region->deviceAddress);
    if (!buffer) {
        return command;
    }

    ResourceAccessCommand::BufferAccess access;
    access.buffer = buffer;
    access.access_index = SYNC_RAY_TRACING_SHADER_SHADER_BINDING_TABLE_READ;
    const VkDeviceSize offset = sbt_address_region->deviceAddress - buffer->deviceAddress;
    access.range = MakeRange(*buffer, offset, sbt_address_region->size);
    access.tag_handle = buffer->Handle();
    access.resource_name = std::string(sbt_buffer_label) + " shader binding table ";
    command.accesses.emplace_back(std::move(access));
    return command;
}

bool SyncValidator::PreCallValidateCmdTraceRaysKHR(VkCommandBuffer commandBuffer,
                                                   const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
                                                   const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
                                                   const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
                                                   const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable,
                                                   uint32_t width, uint32_t height, uint32_t depth,
                                                   const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
    command.Append(MakeSbtBufferAccessCommand(pRaygenShaderBindingTable, "raygen"));
    command.Append(MakeSbtBufferAccessCommand(pMissShaderBindingTable, "miss"));
    command.Append(MakeSbtBufferAccessCommand(pHitShaderBindingTable, "hit"));
    command.Append(MakeSbtBufferAccessCommand(pCallableShaderBindingTable, "callable"));
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdTraceRaysKHR(VkCommandBuffer commandBuffer,
                                                  const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
                                                  const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
                                                  const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
                                                  const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable,
                                                  uint32_t width, uint32_t height, uint32_t depth, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
    command.Append(MakeSbtBufferAccessCommand(pRaygenShaderBindingTable, "raygen"));
    command.Append(MakeSbtBufferAccessCommand(pMissShaderBindingTable, "miss"));
    command.Append(MakeSbtBufferAccessCommand(pHitShaderBindingTable, "hit"));
    command.Append(MakeSbtBufferAccessCommand(pCallableShaderBindingTable, "callable"));
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdTraceRaysIndirectKHR(VkCommandBuffer commandBuffer,
                                                           const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
                                                           const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
                                                           const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
                                                           const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable,
                                                           VkDeviceAddress indirectDeviceAddress,
                                                           const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
    command.Append(MakeSbtBufferAccessCommand(pRaygenShaderBindingTable, "raygen"));
    command.Append(MakeSbtBufferAccessCommand(pMissShaderBindingTable, "miss"));
    command.Append(MakeSbtBufferAccessCommand(pHitShaderBindingTable, "hit"));
    command.Append(MakeSbtBufferAccessCommand(pCallableShaderBindingTable, "callable"));
    if (const vvl::Buffer* indirect_buffer = GetSingleBufferFromDeviceAddress(*device_state, indirectDeviceAddress)) {
        command.Append(
            MakeIndirectBufferAccessCommand(sizeof(VkTraceRaysIndirectCommandKHR), indirect_buffer->VkHandle(), 0, 1, 0));
    }
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdTraceRaysIndirectKHR(VkCommandBuffer commandBuffer,
                                                          const VkStridedDeviceAddressRegionKHR* pRaygenShaderBindingTable,
                                                          const VkStridedDeviceAddressRegionKHR* pMissShaderBindingTable,
                                                          const VkStridedDeviceAddressRegionKHR* pHitShaderBindingTable,
                                                          const VkStridedDeviceAddressRegionKHR* pCallableShaderBindingTable,
                                                          VkDeviceAddress indirectDeviceAddress, const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
    command.Append(MakeSbtBufferAccessCommand(pRaygenShaderBindingTable, "raygen"));
    command.Append(MakeSbtBufferAccessCommand(pMissShaderBindingTable, "miss"));
    command.Append(MakeSbtBufferAccessCommand(pHitShaderBindingTable, "hit"));
    command.Append(MakeSbtBufferAccessCommand(pCallableShaderBindingTable, "callable"));
    if (const vvl::Buffer* indirect_buffer = GetSingleBufferFromDeviceAddress(*device_state, indirectDeviceAddress)) {
        command.Append(
            MakeIndirectBufferAccessCommand(sizeof(VkTraceRaysIndirectCommandKHR), indirect_buffer->VkHandle(), 0, 1, 0));
    }
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}

bool SyncValidator::PreCallValidateCmdTraceRaysIndirect2KHR(VkCommandBuffer commandBuffer, VkDeviceAddress indirectDeviceAddress,
                                                            const ErrorObject& error_obj) const {
    if (!syncval_settings.IsRecordTimeValidationEnabled()) {
        return false;
    }
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    const CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
    if (const vvl::Buffer* indirect_buffer = GetSingleBufferFromDeviceAddress(*device_state, indirectDeviceAddress)) {
        command.Append(
            MakeIndirectBufferAccessCommand(sizeof(VkTraceRaysIndirectCommand2KHR), indirect_buffer->VkHandle(), 0, 1, 0));
    }
    return command.Validate(cb_context, error_obj.location);
}

void SyncValidator::PostCallRecordCmdTraceRaysIndirect2KHR(VkCommandBuffer commandBuffer, VkDeviceAddress indirectDeviceAddress,
                                                           const RecordObject& record_obj) {
    auto cb_state = Get<vvl::CommandBuffer>(commandBuffer);
    CommandBufferContext& cb_context = GetCommandBufferContext(*cb_state);
    const ResourceUsageTag tag = cb_context.NextCommandTag(record_obj.location.function);
    auto command = cb_context.MakeDispatchDrawDescriptorAccessCommand(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
    if (const vvl::Buffer* indirect_buffer = GetSingleBufferFromDeviceAddress(*device_state, indirectDeviceAddress)) {
        command.Append(
            MakeIndirectBufferAccessCommand(sizeof(VkTraceRaysIndirectCommand2KHR), indirect_buffer->VkHandle(), 0, 1, 0));
    }
    cb_context.RecordResourceAccesses(tag, std::move(command), syncval_settings.IsRecordTimeValidationEnabled());
}
}  // namespace syncval
