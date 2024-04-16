/*
 * Copyright (c) 2019-2024 Valve Corporation
 * Copyright (c) 2019-2024 LunarG, Inc.
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
#include "sync/sync_commandbuffer.h"
#include "state_tracker/queue_state.h"

struct PresentedImage;
class QueueBatchContext;
struct QueueSubmitCmdState;
class QueueSyncState;
class SyncValidator;

namespace vvl {
class Semaphore;
}  // namespace vvl

struct AcquiredImage {
    std::shared_ptr<const syncval_state::ImageState> image;
    subresource_adapter::ImageRangeGenerator generator;
    ResourceUsageTag present_tag;
    ResourceUsageTag acquire_tag;
    bool Invalid() const;

    AcquiredImage() = default;
    AcquiredImage(const PresentedImage &presented, ResourceUsageTag acq_tag);
};

// Information associated with a semaphore signal
struct SignalInfo {
    SignalInfo() = default;
    SignalInfo(const std::shared_ptr<QueueBatchContext> &batch, const SyncExecScope &exec_scope, uint64_t timeline_value);
    SignalInfo(const PresentedImage &presented, ResourceUsageTag acquire_tag);
    SignalInfo(uint64_t initial_value);

    // Batch from the first scope of the signal. Not null.
    std::shared_ptr<QueueBatchContext> batch;

    // Use the SyncExecScope::valid_accesses for first access scope
    SemaphoreScope first_scope;

    // Swapchain specific signal info.
    // Batch field is the batch of the last present for the acquired image.
    // The AcquiredImage further limits the scope of the resolve operation, and the "barrier" will also
    // be special case (updating "PRESENTED" write with "ACQUIRE" read, as well as setting the barrier).
    //
    // NOTE: shared_ptr is used here as a memory saver. AcquiredImage is 224 bytes at the time
    // of writing and is not used in the queue submit signals. If we optimize ImageRangeGenerator
    // memory usage then shared_ptr can be replaced by std::optional to avoid allocation.
    std::shared_ptr<AcquiredImage> acquired_image;

    // For timeline semaphores.
    uint64_t timeline_value = 0;
};

// Globally tracks signaled semaphores.
using BinarySignals = vvl::unordered_map<VkSemaphore, SignalInfo>;

using TimelineSignals = vvl::unordered_map<VkSemaphore, std::vector<SignalInfo>>;

// The list of changes that should to be applied to SignaledSemaphores.
// These changes are collected during validation phase of QueueSubmit and are applied in the record phase.
struct SignaledSemaphoresUpdate {
    vvl::unordered_map<VkSemaphore, SignalInfo> signals_to_add;
    vvl::unordered_set<VkSemaphore> signals_to_remove;

    vvl::unordered_map<VkSemaphore, std::vector<SignalInfo>> timeline_signals;

    void OnSignal(const vvl::Semaphore &semaphore_state, const std::shared_ptr<QueueBatchContext> &batch,
                  const VkSemaphoreSubmitInfo &signal_info);
    std::optional<SignalInfo> OnBinaryWait(VkSemaphore semaphore);
    std::optional<SignalInfo> OnTimelineWait(VkSemaphore semaphore, uint64_t wait_value);

    SignaledSemaphoresUpdate(const SyncValidator &sync_validator) : sync_validator_(sync_validator) {}

  private:
    const SyncValidator &sync_validator_;
};

struct FenceSyncState {
    std::shared_ptr<const vvl::Fence> fence;
    ResourceUsageTag tag;
    QueueId queue_id;
    AcquiredImage acquired;  // Iff queue == invalid and acquired.image valid.
    FenceSyncState();
    FenceSyncState(const FenceSyncState &other) = default;
    FenceSyncState(FenceSyncState &&other) = default;
    FenceSyncState &operator=(const FenceSyncState &other) = default;
    FenceSyncState &operator=(FenceSyncState &&other) = default;

    FenceSyncState(const std::shared_ptr<const vvl::Fence> &fence_, QueueId queue_id_, ResourceUsageTag tag_);
    FenceSyncState(const std::shared_ptr<const vvl::Fence> &fence_, const PresentedImage &image, ResourceUsageTag tag_);
};

struct PresentedImageRecord {
    ResourceUsageTag tag;  // the global tag at presentation
    uint32_t image_index;
    uint32_t present_index;
    std::weak_ptr<const syncval_state::Swapchain> swapchain_state;
    std::shared_ptr<const syncval_state::ImageState> image;
};

struct PresentedImage : public PresentedImageRecord {
    std::shared_ptr<QueueBatchContext> batch;
    subresource_adapter::ImageRangeGenerator range_gen;

    PresentedImage() = default;
    void UpdateMemoryAccess(SyncStageAccessIndex usage, ResourceUsageTag tag, AccessContext &access_context) const;
    PresentedImage(const SyncValidator &sync_state, std::shared_ptr<QueueBatchContext> batch, VkSwapchainKHR swapchain,
                   uint32_t image_index, uint32_t present_index, ResourceUsageTag present_tag_);
    // For non-previsously presented images..
    PresentedImage(std::shared_ptr<const syncval_state::Swapchain> swapchain, uint32_t at_index);
    bool Invalid() const;
    void ExportToSwapchain(SyncValidator &);
    void SetImage(uint32_t at_index);
};
using PresentedImages = std::vector<PresentedImage>;

// Store references to ResourceUsageRecords with global tag range within a batch
class BatchAccessLog {
  public:
    struct BatchRecord {
        const QueueSyncState *queue = nullptr;
        uint64_t submit_index = 0;
        uint32_t batch_index = 0;
        uint32_t cb_index = 0;
        ResourceUsageTag base_tag = 0;
    };

    struct AccessRecord {
        const BatchRecord *batch;
        const ResourceUsageRecord *record;
        const DebugNameProvider *debug_name_provider;
        bool IsValid() const { return batch && record; }
    };

    struct CBSubmitLog : DebugNameProvider {
      public:
        CBSubmitLog() = default;
        CBSubmitLog(const CBSubmitLog &batch) = default;
        CBSubmitLog(CBSubmitLog &&other) = default;
        CBSubmitLog &operator=(const CBSubmitLog &other) = default;
        CBSubmitLog &operator=(CBSubmitLog &&other) = default;
        CBSubmitLog(const BatchRecord &batch, std::shared_ptr<const CommandExecutionContext::CommandBufferSet> cbs,
                    std::shared_ptr<const CommandExecutionContext::AccessLog> log);
        CBSubmitLog(const BatchRecord &batch, const CommandBufferAccessContext &cb,
                    const std::vector<std::string> &initial_label_stack);
        size_t Size() const { return log_->size(); }
        AccessRecord operator[](ResourceUsageTag tag) const;

        // DebugNameProvider
        std::string GetDebugRegionName(const ResourceUsageRecord &record) const override;

      private:
        BatchRecord batch_;
        std::shared_ptr<const CommandExecutionContext::CommandBufferSet> cbs_;
        std::shared_ptr<const CommandExecutionContext::AccessLog> log_;
        // label stack at the point when command buffer is submitted to the queue
        std::vector<std::string> initial_label_stack_;

        // TODO: remove this field and use (*cbs_)[0]->GetLabelCommands() directly
        // when timeline semaphore support is implemented.
        //
        // Until then, there is no guarantee command buffers stored in cbs_ are what
        // they are supposed to be when timeline semaphores are used (they can be reused
        // after wait on timeline semaphore). When this happens, validation might report
        // false positives (which is okay for unsupported feeature), but label code can crash.
        // Make a copy of label commands as a temporary protection measure.
        std::vector<vvl::CommandBuffer::LabelCommand> label_commands_;
    };

    void Import(const BatchRecord &batch, const CommandBufferAccessContext &cb_access,
                const std::vector<std::string> &initial_label_stack);
    void Import(const BatchAccessLog &other);
    void Insert(const BatchRecord &batch, const ResourceUsageRange &range,
                std::shared_ptr<const CommandExecutionContext::AccessLog> log);

    void Trim(const ResourceUsageTagSet &used);
    // AccessRecord lookup is based on global tags
    AccessRecord operator[](ResourceUsageTag tag) const;
    BatchAccessLog() {}

  private:
    using CBSubmitLogRangeMap = sparse_container::range_map<ResourceUsageTag, CBSubmitLog>;
    CBSubmitLogRangeMap log_map_;
};

class QueueBatchContext;
using BatchContextPtr = std::shared_ptr<QueueBatchContext>;
using BatchContextConstPtr = std::shared_ptr<const QueueBatchContext>;

struct CommandBufferInfo {
    uint32_t index = 0;
    std::shared_ptr<const syncval_state::CommandBuffer> cb_state;
    CommandBufferInfo(uint32_t index, std::shared_ptr<const syncval_state::CommandBuffer> &&cb_state)
        : index(index), cb_state(std::move(cb_state)) {}
};

struct UnresolvedBatch {
    BatchContextPtr batch;

    // Waits-before-signals that prevent this batch to be resolved
    std::vector<VkSemaphoreSubmitInfo> unresolved_waits;

    // Batches from resolved waits
    std::vector<BatchContextConstPtr> resolved_batches;

    std::vector<CommandBufferInfo> command_buffers;

    std::vector<VkSemaphoreSubmitInfo> signals;

    // Queue's label stack at the beginning of this batch
    std::vector<std::string> label_stack;
};

class QueueBatchContext : public CommandExecutionContext, public std::enable_shared_from_this<QueueBatchContext> {
  public:
    class PresentResourceRecord : public AlternateResourceUsage::RecordBase {
      public:
        using Base_ = AlternateResourceUsage::RecordBase;
        Base_::Record MakeRecord() const override;
        ~PresentResourceRecord() override {}
        PresentResourceRecord(const PresentedImageRecord &presented) : presented_(presented) {}
        std::ostream &Format(std::ostream &out, const SyncValidator &sync_state) const override;

      private:
        PresentedImageRecord presented_;
    };

    class AcquireResourceRecord : public AlternateResourceUsage::RecordBase {
      public:
        using Base_ = AlternateResourceUsage::RecordBase;
        Base_::Record MakeRecord() const override;
        AcquireResourceRecord(const PresentedImageRecord &presented, ResourceUsageTag tag, vvl::Func command)
            : presented_(presented), acquire_tag_(tag), command_(command) {}
        std::ostream &Format(std::ostream &out, const SyncValidator &sync_state) const override;

      private:
        PresentedImageRecord presented_;
        ResourceUsageTag acquire_tag_;
        vvl::Func command_;
    };

    using Ptr = std::shared_ptr<QueueBatchContext>;
    using ConstPtr = std::shared_ptr<const QueueBatchContext>;

    struct ResolveWaitsResult {
        std::vector<ConstPtr> resolved_batches;
        std::vector<VkSemaphoreSubmitInfo> unresolved_waits;  // waits before signals
    };

    QueueBatchContext(const SyncValidator &sync_state, const QueueSyncState &queue_state);
    QueueBatchContext(const SyncValidator &sync_state);
    QueueBatchContext() = delete;
    QueueBatchContext &operator=(QueueBatchContext &&other);
    void Trim();
    void ResetAccessContext() { access_context_.Reset(); }

    std::string FormatUsage(ResourceUsageTag tag) const override;
    AccessContext *GetCurrentAccessContext() override { return current_access_context_; }
    const AccessContext *GetCurrentAccessContext() const override { return current_access_context_; }
    SyncEventsContext *GetCurrentEventsContext() override { return &events_context_; }
    const SyncEventsContext *GetCurrentEventsContext() const override { return &events_context_; }
    const QueueSyncState *GetQueueSyncState() { return queue_state_; }
    VkQueueFlags GetQueueFlags() const;
    QueueId GetQueueId() const override;
    ExecutionType Type() const override { return kSubmitted; }

    ResourceUsageTag SetupBatchTags(uint32_t tag_count);
    void ResetEventsContext() { events_context_.Clear(); }

    // For Submit
    ResolveWaitsResult ResolveSubmitWaits(vvl::span<const VkSemaphoreSubmitInfo> wait_semaphores,
                                          SignaledSemaphoresUpdate &signaled_semaphores_update);
    bool ValidateSubmit(const std::vector<CommandBufferInfo> &command_buffers, uint64_t submit_index, uint32_t batch_index,
                        std::vector<std::string> &current_label_stack, const ErrorObject &error_obj);
    std::vector<CommandBufferInfo> GetCommandBuffers(const VkSubmitInfo2 &submit_info);
    void ResolveSubmittedCommandBuffer(const AccessContext &recorded_context, ResourceUsageTag offset);

    // For Present
    std::vector<ConstPtr> ResolvePresentWaits(vvl::span<const VkSemaphore> wait_semaphores, const PresentedImages &presented_images,
                                              SignaledSemaphoresUpdate &signaled_semaphores_update);
    bool DoQueuePresentValidate(const Location &loc, const PresentedImages &presented_images);
    void DoPresentOperations(const PresentedImages &presented_images);
    void LogPresentOperations(const PresentedImages &presented_images, uint64_t submit_index);

    // For Acquire
    void SetupAccessContext(const PresentedImage &presented);
    void DoAcquireOperation(const PresentedImage &presented);
    void LogAcquireOperation(const PresentedImage &presented, vvl::Func command);

    VulkanTypedHandle Handle() const override;

    template <typename Predicate>
    void ApplyPredicatedWait(Predicate &predicate);
    void ApplyTaggedWait(QueueId queue_id, ResourceUsageTag tag);
    void ApplyAcquireWait(const AcquiredImage &acquired);

    void BeginRenderPassReplaySetup(ReplayState &replay, const SyncOpBeginRenderPass &begin_op) override;
    void NextSubpassReplaySetup(ReplayState &replay) override;
    void EndRenderPassReplayCleanup(ReplayState &replay) override;

    [[nodiscard]] std::vector<ConstPtr> RegisterAsyncContexts(const std::vector<ConstPtr> &batches_resolved);
    void ResolveLastBatch(const QueueBatchContext::ConstPtr &last_batch);

    void ResolveSubmitSemaphoreWait(const SignalInfo &signal_info, uint64_t timeline_wait_value, VkPipelineStageFlags2 wait_mask);
    void ImportTags(const QueueBatchContext &from);

  private:
    void ResolvePresentSemaphoreWait(const SignalInfo &signal_info, const PresentedImages &presented_images);

  public:
    const QueueSyncState *queue_state_ = nullptr;
    ResourceUsageRange tag_range_ = ResourceUsageRange(0, 0);  // Range of tags referenced by cbs_referenced

    AccessContext access_context_;
    AccessContext *current_access_context_;
    SyncEventsContext events_context_;
    BatchAccessLog batch_log_;
    std::vector<ResourceUsageTag> queue_sync_tag_;
};

class QueueSyncState {
  public:
    QueueSyncState(QueueId id, const std::shared_ptr<vvl::Queue> &queue_state) : id_(id), queue_state_(queue_state) {}

    VulkanTypedHandle Handle() const { return queue_state_->Handle(); }
    QueueBatchContext::ConstPtr LastBatch() const { return last_batch_; }
    QueueBatchContext::Ptr LastBatch() { return last_batch_; }
    void UpdateLastBatch();
    void UpdateUnresolvedBatches();
    const vvl::Queue *GetQueueState() const { return queue_state_.get(); }
    VkQueueFlags GetQueueFlags() const { return queue_state_->queueFamilyProperties.queueFlags; }
    QueueId GetQueueId() const { return id_; }

    // Method is const but updates mutable sumbit_index atomically.
    uint64_t ReserveSubmitId() const;

    // Method is const but updates mutable pending_last_batch and relies on the queue external synchronization
    void SetPendingLastBatch(QueueBatchContext::Ptr &&last) const;

    QueueBatchContext::Ptr PendingLastBatch() const { return pending_last_batch_; }

    void SetPendingTimelineSignalPropagationQueues(std::vector<std::shared_ptr<QueueSyncState>> &&pending_queues) const {
        pending_timeline_propagation_queues_ = std::move(pending_queues);
    }
    void UpdatePendingTimelineSignalPropagationQueues() {
        for (auto &queue : pending_timeline_propagation_queues_) {
            queue->UpdateLastBatch();
        }
        pending_timeline_propagation_queues_.resize(0);
    }

    bool HasUnresolvedBatches() const { return !unresolved_batches_.empty(); }
    void SetPendingUnresolvedBatches(std::vector<UnresolvedBatch> &&unresolved_batches) const {
        pending_unresolved_batches_ = std::move(unresolved_batches);
        apply_pending_unresolved_batches_ = true;
    }
    const std::vector<UnresolvedBatch> GetUnresolvedBatches() const { return unresolved_batches_; }
    std::vector<UnresolvedBatch> &GetPendingUnresolvedBatches() { return pending_unresolved_batches_; }

  private:
    const QueueId id_;
    std::shared_ptr<vvl::Queue> queue_state_;

    mutable std::atomic<uint64_t> submit_index_ = 0;

    mutable QueueBatchContext::Ptr pending_last_batch_;
    mutable std::vector<std::shared_ptr<QueueSyncState>> pending_timeline_propagation_queues_;
    QueueBatchContext::Ptr last_batch_;

    // Batches collected during validation phase to be applied in the record phase.
    // Need a separate flag because we might need to apply an empty list too
    // (e.g. to reset existing unresolved batches in the case of validation error).
    mutable bool apply_pending_unresolved_batches_ = false;
    mutable std::vector<UnresolvedBatch> pending_unresolved_batches_;

    // The first batch in this list is always due to wait-before-signal dependency.
    // After we found wait-before-signal batch, we must also put queue's all subsequent
    // batches in this list because they can't be resolved before the first batch is
    // resolved (respect submission order).
    //
    // When the first batch is resolved we continue with resolving other queued batches
    // until we reach the batch that has unresolved wait-before-signal (and it becomes
    // the new first element of the list) or all unresolved batches are processed.
    std::vector<UnresolvedBatch> unresolved_batches_;
};

// The converter needs to be more complex than simply an array of VkSubmitInfo2 structures.
// In order to convert from Info->Info2, arrays of VkSemaphoreSubmitInfo and VkCommandBufferSubmitInfo
// structures must be created for the pWaitSemaphoreInfos, pCommandBufferInfos, and pSignalSemaphoreInfos
// which comprise the converted VkSubmitInfo information. The created VkSubmitInfo2 structure then references the storage
// of the arrays, which must have a lifespan longer than the conversion, s.t. the ensuing valdation/record operations
// can reference them.  The resulting VkSubmitInfo2 is then copied into an additional which takes the place of the pSubmits
// parameter.
struct SubmitInfoConverter {
    struct BatchStore {
        BatchStore(const VkSubmitInfo &info, VkQueueFlags queue_flags);

        static VkSemaphoreSubmitInfo WaitSemaphore(const VkSubmitInfo &info, uint32_t index);
        static VkCommandBufferSubmitInfo CommandBuffer(const VkSubmitInfo &info, uint32_t index);
        static VkSemaphoreSubmitInfo SignalSemaphore(const VkSubmitInfo &info, uint32_t index, VkQueueFlags queue_flags);

        std::vector<VkSemaphoreSubmitInfo> waits;
        std::vector<VkCommandBufferSubmitInfo> cbs;
        std::vector<VkSemaphoreSubmitInfo> signals;
        VkSubmitInfo2 info2;
    };

    SubmitInfoConverter(uint32_t count, const VkSubmitInfo *infos, VkQueueFlags queue_flags);

    std::vector<BatchStore> info_store;
    std::vector<VkSubmitInfo2> info2s;
};

struct QueueSubmitCmdState {
    std::shared_ptr<const QueueSyncState> queue;
    SignaledSemaphoresUpdate signaled_semaphores_update;
    QueueSubmitCmdState(const SyncValidator &sync_validator) : signaled_semaphores_update(sync_validator) {}
};
