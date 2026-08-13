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
#pragma once

#include "sync/sync_access_context.h"
#include "sync/sync_barrier.h"
#include "error_message/error_location.h"

#include <utility>
#include <variant>

namespace vvl {
class Event;
class ImageView;
class RenderPass;
}  // namespace vvl

namespace syncval {
class CommandBufferContext;
struct SyncEnvironment;
class RenderPassAccessContext;

// Replay actions store the data needed to update synchronization state. Hazard detection and
// render pass traversal are intentionally separate.
struct PipelineBarrierReplayAction {
    PipelineBarrierReplayAction(BarrierSet&& barrier_set);

    BarrierSet barrier_set;
};

struct SetEventReplayAction {
    SetEventReplayAction(std::shared_ptr<const vvl::Event>&& event, const SyncExecScope& src_exec_scope,
                         std::shared_ptr<const AccessContext>&& src_access_context, const Location& loc);

    vvl::Func command = vvl::Func::Empty;
    std::shared_ptr<const vvl::Event> event;
    // Snapshot of the command buffer's access context at set event time
    std::shared_ptr<const AccessContext> recorded_context;
    SyncExecScope src_exec_scope;
};

struct ResetEventReplayAction {
    ResetEventReplayAction(std::shared_ptr<const vvl::Event>&& event, const SyncExecScope& exec_scope, const Location& loc);

    vvl::Func command = vvl::Func::Empty;
    std::shared_ptr<const vvl::Event> event;
    SyncExecScope exec_scope;
};

struct WaitEventsReplayAction {
    WaitEventsReplayAction(std::vector<std::shared_ptr<const vvl::Event>>&& events, std::vector<BarrierSet>&& barrier_sets,
                           const Location& loc);

    vvl::Func command = vvl::Func::Empty;
    std::vector<std::shared_ptr<const vvl::Event>> events;
    std::vector<BarrierSet> barrier_sets;
};

// Render pass traversal is ordered with replay actions, but it is not itself a synchronization action.
struct RenderPassReplayTraversal {
    enum class Type {
        kBegin,
        kNextSubpass,
        kEnd,
    };

    RenderPassReplayTraversal(std::shared_ptr<const vvl::RenderPass>&& rp_state,
                              std::vector<std::shared_ptr<const vvl::ImageView>>&& attachments,
                              const RenderPassAccessContext* rp_context)
        : type(Type::kBegin), rp_state(std::move(rp_state)), attachments(std::move(attachments)), rp_context(rp_context) {}
    explicit RenderPassReplayTraversal(Type type) : type(type) {}

    // Keep references to rp_state and attachments in case they are deleted.
    // The RenderPassAccessContext keeps only pointers to them.
    Type type;
    std::shared_ptr<const vvl::RenderPass> rp_state;
    std::vector<std::shared_ptr<const vvl::ImageView>> attachments;
    const RenderPassAccessContext* rp_context = nullptr;
};

using ReplayOperation = std::variant<PipelineBarrierReplayAction, SetEventReplayAction, ResetEventReplayAction,
                                     WaitEventsReplayAction, RenderPassReplayTraversal>;

void ApplyReplayAction(const ReplayOperation& operation, SyncEnvironment& env, AccessContext& access_context,
                       ResourceUsageTag exec_tag);

// Most replay entries are boundaries between ranges of recorded first accesses. Entries that
// themselves record an access (currently layout transitions) request an additional check at their tag.
enum class FirstUseCheck {
    kNone,
    kCurrentTag,
};

struct ReplayEntry {
    template <typename Operation>
    ReplayEntry(ResourceUsageTag tag, FirstUseCheck first_use_check, Operation&& operation)
        : tag(tag), first_use_check(first_use_check), operation(std::forward<Operation>(operation)) {}

    ResourceUsageTag tag = 0;
    FirstUseCheck first_use_check = FirstUseCheck::kNone;
    ReplayOperation operation;
};

bool ValidateFirstUseHazards(SyncEnvironment& env, AccessContext& destination_context,
                             const CommandBufferContext& recorded_cb_context, ResourceUsageTag base_tag, const Location& cb_loc);

}  // namespace syncval
