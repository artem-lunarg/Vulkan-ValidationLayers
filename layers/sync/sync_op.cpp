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

using vvl::Func;

namespace syncval {

PipelineBarrierReplayAction::PipelineBarrierReplayAction(BarrierSet&& barrier_set) : barrier_set(std::move(barrier_set)) {}

SetEventReplayAction::SetEventReplayAction(std::shared_ptr<const vvl::Event>&& event, const SyncExecScope& src_exec_scope,
                                           std::shared_ptr<const AccessContext>&& src_access_context, const Location& loc)
    : command(loc.function),
      event(std::move(event)),
      recorded_context(std::move(src_access_context)),
      src_exec_scope(src_exec_scope) {}

ResetEventReplayAction::ResetEventReplayAction(std::shared_ptr<const vvl::Event>&& event, const SyncExecScope& exec_scope,
                                               const Location& loc)
    : command(loc.function), event(std::move(event)), exec_scope(exec_scope) {}

WaitEventsReplayAction::WaitEventsReplayAction(std::vector<std::shared_ptr<const vvl::Event>>&& events,
                                               std::vector<BarrierSet>&& barrier_sets, const Location& loc)
    : command(loc.function), events(std::move(events)), barrier_sets(std::move(barrier_sets)) {}

void ApplyReplayAction(const ReplayOperation& operation, SyncEnvironment& env, AccessContext& access_context,
                       ResourceUsageTag exec_tag) {
    if (const auto* barrier = std::get_if<PipelineBarrierReplayAction>(&operation)) {
        ApplyBarrier(env, access_context, barrier->barrier_set, exec_tag);
    } else if (const auto* set_event = std::get_if<SetEventReplayAction>(&operation)) {
        // Merge the current context with the snapshot recorded by set event without modifying the snapshot.
        auto merged_context = std::make_shared<AccessContext>(*access_context.validator);
        merged_context->InitFrom(access_context);
        merged_context->ResolveFromContext(QueueTagOffsetBarrierAction(env.queue_id, exec_tag), *set_event->recorded_context);
        merged_context->TrimAndClearFirstAccess();

        ApplyCmdSetEvent(env, set_event->event, set_event->src_exec_scope, merged_context, exec_tag, set_event->command);
    } else if (const auto* reset_event = std::get_if<ResetEventReplayAction>(&operation)) {
        ApplyCmdResetEvent(env, reset_event->event, exec_tag, reset_event->command);
    } else if (const auto* wait_events = std::get_if<WaitEventsReplayAction>(&operation)) {
        ApplyCmdWaitEvents(env, access_context, wait_events->events, wait_events->barrier_sets, exec_tag, wait_events->command);
    }
}

namespace {

// Provides the recorded and destination contexts currently selected by replay traversal.
// Command buffer contexts are selected outside a render pass and subpass contexts inside one.
class ReplayContexts {
  public:
    ReplayContexts(const CommandBufferContext& recorded_cb_context, AccessContext& destination_context, VkQueueFlags queue_flags)
        : recorded_cb_context_(recorded_cb_context), destination_context_(destination_context), queue_flags_(queue_flags) {}

    void BeginRenderPass(const RenderPassAccessContext& rp_context) {
        const vvl::RenderPass& render_pass = *rp_context.GetRenderPassState();
        const uint32_t subpass_count = render_pass.create_info.subpassCount;

        rp_context_ = &rp_context;
        current_subpass_ = 0;
        destination_subpass_contexts_ = InitSubpassContexts(queue_flags_, render_pass, destination_context_);

        // Replace the async contexts with the async context of the destination context. For replay
        // we don't care about async subpasses, only async queue batches.
        for (uint32_t i = 0; i < subpass_count; i++) {
            AccessContext& subpass_context = destination_subpass_contexts_[i];
            subpass_context.ClearAsyncContexts();
            subpass_context.ImportAsyncContexts(destination_context_);
        }
    }

    void NextSubpass() {
        assert(rp_context_);
        const uint32_t subpass_count = rp_context_->GetRenderPassState()->create_info.subpassCount;
        if (current_subpass_ + 1 < subpass_count) {
            // Any store/resolve operations happen before the NextSubpass tag.
            current_subpass_++;
        }
    }

    void EndRenderPass() {
        assert(rp_context_);
        const uint32_t subpass_count = rp_context_->GetRenderPassState()->create_info.subpassCount;
        destination_context_.ResolveChildContexts(vvl::make_span(destination_subpass_contexts_.get(), subpass_count));
        rp_context_ = nullptr;
        current_subpass_ = 0;
        destination_subpass_contexts_.reset();
    }

    const AccessContext& GetRecordedContext() const {
        return rp_context_ ? rp_context_->GetSubpassContexts()[current_subpass_] : recorded_cb_context_.GetCbAccessContext();
    }

    AccessContext& GetDestinationContext() {
        return rp_context_ ? destination_subpass_contexts_[current_subpass_] : destination_context_;
    }

  private:
    const CommandBufferContext& recorded_cb_context_;
    AccessContext& destination_context_;
    const VkQueueFlags queue_flags_;

    const RenderPassAccessContext* rp_context_ = nullptr;
    uint32_t current_subpass_ = 0;

    // Unlike the recorded subpass contexts, these contain no recorded accesses. They only
    // store subpass dependencies applied to the destination state.
    std::unique_ptr<AccessContext[]> destination_subpass_contexts_;
};

// This dispatcher is intentionally isolated so it can be removed when event command validation moves to core validation.
bool ValidateEventCommand(const ReplayOperation& operation, const SyncEnvironment& env, ResourceUsageTag exec_tag) {
    if (const auto* set_event = std::get_if<SetEventReplayAction>(&operation)) {
        return ValidateCmdSetEvent(env, set_event->event, set_event->src_exec_scope, exec_tag, Location(set_event->command));
    }
    if (const auto* reset_event = std::get_if<ResetEventReplayAction>(&operation)) {
        return ValidateCmdResetEvent(env, reset_event->event, reset_event->exec_scope, exec_tag, Location(reset_event->command));
    }
    if (const auto* wait_events = std::get_if<WaitEventsReplayAction>(&operation)) {
        return ValidateCmdWaitEventsEventState(env, wait_events->events, exec_tag, Location(wait_events->command));
    }
    return false;
}

// Unlike the event command checks above, an image layout transition is a memory access and remains part of syncval.
bool DetectReplayActionHazard(const ReplayOperation& operation, const SyncEnvironment& env,
                              const AccessContext& destination_context, ResourceUsageTag exec_tag) {
    const auto* wait_events = std::get_if<WaitEventsReplayAction>(&operation);
    return wait_events ? DetectCmdWaitEventsImageBarrierHazard(env, destination_context, wait_events->events,
                                                               wait_events->barrier_sets, exec_tag, Location(wait_events->command))
                       : false;
}

void ApplyReplayTraversal(const RenderPassReplayTraversal& traversal, ReplayContexts& contexts) {
    switch (traversal.type) {
        case RenderPassReplayTraversal::Type::kBegin:
            contexts.BeginRenderPass(*traversal.rp_context);
            break;

        case RenderPassReplayTraversal::Type::kNextSubpass:
            contexts.NextSubpass();
            break;

        case RenderPassReplayTraversal::Type::kEnd:
            contexts.EndRenderPass();
            break;
    }
}

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
        // Validate all first accesses until the next replay entry.
        first_use_range.end = entry.tag;
        skip |= DetectFirstUseHazard(env, recorded_cb_context, contexts, first_use_range, cb_loc);

        // Traversal selects the recorded and destination contexts used by checks at this tag.
        const auto* traversal = std::get_if<RenderPassReplayTraversal>(&entry.operation);
        if (traversal) ApplyReplayTraversal(*traversal, contexts);

        if (entry.first_use_check == FirstUseCheck::kCurrentTag) {
            // The replay entry itself records a first access (currently an image layout transition).
            skip |= DetectFirstUseHazard(env, recorded_cb_context, contexts, {entry.tag, entry.tag + 1}, cb_loc);
        }

        if (!traversal) {
            const ResourceUsageTag exec_tag = base_tag + entry.tag;
            // Temporary event command checks are deliberately separate from permanent syncval hazard detection.
            skip |= ValidateEventCommand(entry.operation, env, exec_tag);
            skip |= DetectReplayActionHazard(entry.operation, env, contexts.GetDestinationContext(), exec_tag);
            ApplyReplayAction(entry.operation, env, contexts.GetDestinationContext(), exec_tag);
        }

        // Advance past the replay entry.
        first_use_range.begin = entry.tag + 1;
    }

    // Validate first accesses after the last replay entry.
    first_use_range.end = ResourceUsageRecord::kMaxIndex;
    skip |= DetectFirstUseHazard(env, recorded_cb_context, contexts, first_use_range, cb_loc);
    return skip;
}

}  // namespace syncval
