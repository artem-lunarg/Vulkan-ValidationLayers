/*
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

#include "sync/sync_op.h"
#include "sync/sync_render_pass.h"
#include "sync/sync_command_buffer.h"
#include "sync/sync_event.h"
#include "sync/sync_image.h"
#include "sync/sync_validation.h"

#include "state_tracker/buffer_state.h"
#include "state_tracker/event_state.h"
#include "state_tracker/render_pass_state.h"

#include "utils/image_utils.h"
#include "utils/sync_utils.h"

namespace syncval {

PipelineBarrierAction::PipelineBarrierAction(BarrierSet&& barrier_set) : barrier_set_(std::move(barrier_set)) {}

bool PipelineBarrierAction::Validate(const SyncEnvironment& /*env*/, const AccessContext& /*destination_context*/,
                                     ResourceUsageTag /*exec_tag*/) const {
    // Barrier layout transitions are validated as first accesses (ReplayEntry::first_use_check)
    return false;
}

void PipelineBarrierAction::Apply(SyncEnvironment& env, AccessContext& access_context, const ResourceUsageTag exec_tag) const {
    ApplyBarrier(env, access_context, barrier_set_, exec_tag);
}

SetEventAction::SetEventAction(std::shared_ptr<const vvl::Event>&& event, const SyncExecScope& src_exec_scope,
                               std::shared_ptr<const AccessContext>&& src_access_context, const Location& loc)
    : command_(loc.function),
      event_(std::move(event)),
      recorded_context_(std::move(src_access_context)),
      src_exec_scope_(src_exec_scope) {}

bool SetEventAction::Validate(const SyncEnvironment& env, const AccessContext& /*destination_context*/,
                              ResourceUsageTag exec_tag) const {
    // Temporary event command state check. Planned to move to core validation.
    return ValidateCmdSetEvent(env, event_, src_exec_scope_, exec_tag, Location(command_));
}

void SetEventAction::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag exec_tag) const {
    // Create a copy of the current context, and merge in the state snapshot at record set event time
    // Note: we mustn't change the recorded context copy, as a given CB could be submitted more than once (in generaL)

    // Note: merged_context is a copy of the access_context, combined with the recorded context
    auto merged_context = std::make_shared<AccessContext>(*access_context.validator);
    merged_context->InitFrom(access_context);
    merged_context->ResolveFromContext(QueueTagOffsetBarrierAction(env.queue_id, exec_tag), *recorded_context_);
    merged_context->TrimAndClearFirstAccess();  // Ensure the copy is minimal and normalized

    ApplyCmdSetEvent(env, event_, src_exec_scope_, merged_context, exec_tag, command_);
}

ResetEventAction::ResetEventAction(std::shared_ptr<const vvl::Event>&& event, const SyncExecScope& exec_scope, const Location& loc)
    : command_(loc.function), event_(std::move(event)), exec_scope_(exec_scope) {}

bool ResetEventAction::Validate(const SyncEnvironment& env, const AccessContext& /*destination_context*/,
                                ResourceUsageTag exec_tag) const {
    // Temporary event command state check. Planned to move to core validation.
    return ValidateCmdResetEvent(env, event_, exec_scope_, exec_tag, Location(command_));
}

void ResetEventAction::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag exec_tag) const {
    ApplyCmdResetEvent(env, event_, exec_tag, command_);
}

WaitEventsAction::WaitEventsAction(std::vector<std::shared_ptr<const vvl::Event>>&& events, std::vector<BarrierSet>&& barrier_sets,
                                   const Location& loc)
    : command_(loc.function), events_(std::move(events)), barrier_sets_(std::move(barrier_sets)) {}

bool WaitEventsAction::Validate(const SyncEnvironment& env, const AccessContext& destination_context,
                                ResourceUsageTag exec_tag) const {
    // Temporary event command state check. Planned to move to core validation.
    bool skip = ValidateCmdWaitEventsEventState(env, events_, exec_tag, Location(command_));

    // Image layout transitions are memory accesses and remain permanent syncval functionality
    skip |= DetectCmdWaitEventsImageBarrierHazard(env, destination_context, events_, barrier_sets_, exec_tag, Location(command_));
    return skip;
}

void WaitEventsAction::Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag exec_tag) const {
    ApplyCmdWaitEvents(env, access_context, events_, barrier_sets_, exec_tag, command_);
}

namespace {

// Tracks the current pair of contexts that first-use validation checks against: recorded first
// accesses come from the recorded context, replayed synchronization state accumulates in the
// destination context. These are the command buffer level contexts, except within a render pass,
// where each subpass has its own recorded and destination contexts.
class ReplayContexts {
  public:
    ReplayContexts(const CommandBufferContext& recorded_cb_context, AccessContext& destination_context, VkQueueFlags queue_flags)
        : recorded_cb_context_(recorded_cb_context), destination_context_(destination_context), queue_flags_(queue_flags) {}

    void Update(ContextUpdate context_update, const RenderPassAccessContext* rp_context) {
        switch (context_update) {
            case ContextUpdate::kNone:
                break;
            case ContextUpdate::kBeginRenderPass:
                BeginRenderPass(*rp_context);
                break;
            case ContextUpdate::kNextSubpass:
                NextSubpass();
                break;
            case ContextUpdate::kEndRenderPass:
                EndRenderPass();
                break;
        }
    }

    const AccessContext& GetRecordedContext() const {
        return rp_context_ ? rp_context_->GetSubpassContexts()[current_subpass_] : recorded_cb_context_.GetCbAccessContext();
    }

    AccessContext& GetDestinationContext() {
        return rp_context_ ? destination_subpass_contexts_[current_subpass_] : destination_context_;
    }

  private:
    void BeginRenderPass(const RenderPassAccessContext& rp_context) {
        const vvl::RenderPass& render_pass = *rp_context.GetRenderPassState();
        rp_context_ = &rp_context;
        current_subpass_ = 0;
        destination_subpass_contexts_ = InitSubpassContexts(queue_flags_, render_pass, destination_context_);

        // Replace the async contexts with the async context of the destination context. For replay
        // we don't care about async subpasses, only async queue batches.
        for (uint32_t i = 0; i < render_pass.create_info.subpassCount; i++) {
            AccessContext& subpass_context = destination_subpass_contexts_[i];
            subpass_context.ClearAsyncContexts();
            subpass_context.ImportAsyncContexts(destination_context_);
        }
    }

    void NextSubpass() {
        assert(rp_context_);
        // Store and resolve operations happen before the NextSubpass tag, so the current subpass is done
        const uint32_t subpass_count = rp_context_->GetRenderPassState()->create_info.subpassCount;
        if (current_subpass_ + 1 < subpass_count) {
            current_subpass_++;
        }
    }

    void EndRenderPass() {
        assert(rp_context_);
        // Store and resolve operations happen before the EndRenderPass tag. The final layout transitions
        // are recorded in the command buffer context (not the render pass context), so switching back to
        // the command buffer contexts here lets the check at the EndRenderPass tag validate them.
        const uint32_t subpass_count = rp_context_->GetRenderPassState()->create_info.subpassCount;
        destination_context_.ResolveChildContexts(vvl::make_span(destination_subpass_contexts_.get(), subpass_count));

        rp_context_ = nullptr;
        current_subpass_ = 0;
        destination_subpass_contexts_.reset();
    }

    const CommandBufferContext& recorded_cb_context_;
    AccessContext& destination_context_;
    const VkQueueFlags queue_flags_;

    // Render pass instance state, set between BeginRenderPass and EndRenderPass
    const RenderPassAccessContext* rp_context_ = nullptr;
    uint32_t current_subpass_ = 0;

    // Destination versions of the subpass contexts. Unlike the recorded subpass contexts they
    // contain no recorded accesses, only subpass dependencies applied to the destination state.
    std::unique_ptr<AccessContext[]> destination_subpass_contexts_;
};

bool DetectFirstUseHazard(const SyncEnvironment& env, const CommandBufferContext& recorded_cb_context, ReplayContexts& contexts,
                          const ResourceUsageRange& first_use_range, const Location& cb_loc) {
    if (!first_use_range.non_empty()) {
        return false;
    }
    const HazardResult hazard =
        contexts.GetRecordedContext().DetectFirstUseHazard(env.queue_id, first_use_range, contexts.GetDestinationContext());
    if (hazard.IsHazard()) {
        LogObjectList objlist(env.handle, recorded_cb_context.GetCBState().Handle());
        const std::string error = env.validator.error_messages_.FirstUseError(env, hazard, recorded_cb_context, cb_loc.index);
        return env.validator.SyncError(hazard.Hazard(), objlist, cb_loc, error);
    }
    return false;
}

}  // namespace

// Validate first-use hazards. The following describes how it works.
//
// The first access to a memory location can occur anywhere in the command buffer
// (not necessarily at the beginning), and first accesses to different resources
// may be interleaved with barriers. To validate each first access against accesses
// from previous submissions, we need to replay all barriers that occur before that
// specific first access.
//
// This defines the algorithm: replay barriers until we reach the next first access,
// validate that first access, then continue replaying barriers until the next first
// access, validate that one, and so on until we reach the end of the command buffer.
bool ValidateFirstUseHazards(SyncEnvironment& env, AccessContext& destination_context,
                             const CommandBufferContext& recorded_cb_context, ResourceUsageTag base_tag, const Location& cb_loc) {
    bool skip = false;
    ReplayContexts contexts(recorded_cb_context, destination_context, env.queue_flags);
    ResourceUsageRange first_use_range = {0, 0};

    for (const ReplayEntry& entry : recorded_cb_context.GetReplayEntries()) {
        // Validate first accesses recorded before this entry
        first_use_range.end = entry.tag;
        skip |= DetectFirstUseHazard(env, recorded_cb_context, contexts, first_use_range, cb_loc);

        // Render pass boundaries select the contexts used by the checks below
        contexts.Update(entry.context_update, entry.rp_context);

        if (entry.first_use_check) {
            // The entry itself records a first access (an image layout transition)
            skip |= DetectFirstUseHazard(env, recorded_cb_context, contexts, {entry.tag, entry.tag + 1}, cb_loc);
        }

        if (entry.action) {
            const ResourceUsageTag exec_tag = base_tag + entry.tag;
            skip |= entry.action->Validate(env, contexts.GetDestinationContext(), exec_tag);
            entry.action->Apply(env, contexts.GetDestinationContext(), exec_tag);
        }

        // Continue with first accesses recorded after this entry
        first_use_range.begin = entry.tag + 1;
    }

    // Validate first accesses after the last replay entry
    first_use_range.end = ResourceUsageRecord::kMaxIndex;
    skip |= DetectFirstUseHazard(env, recorded_cb_context, contexts, first_use_range, cb_loc);
    return skip;
}

}  // namespace syncval
