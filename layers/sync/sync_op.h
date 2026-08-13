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

#include <memory>

namespace vvl {
class Event;
class ImageView;
class RenderPass;
}  // namespace vvl

namespace syncval {
class CommandBufferContext;
struct SyncEnvironment;
class RenderPassAccessContext;

// A replay action replays a recorded synchronization command (barrier or event command) when its
// command buffer is executed (vkCmdExecuteCommands) or submitted (vkQueueSubmit): Validate checks
// replay state against the destination context, Apply updates destination synchronization state.
//
// NOTE: the event command state checks behind Validate are temporary and planned to move to core
// validation; the wait events image barrier hazard detection is permanent syncval functionality.
class ReplayAction {
  public:
    virtual ~ReplayAction() = default;
    virtual bool Validate(const SyncEnvironment& env, const AccessContext& destination_context,
                          ResourceUsageTag exec_tag) const = 0;
    virtual void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag exec_tag) const = 0;
};

class PipelineBarrierAction : public ReplayAction {
  public:
    PipelineBarrierAction(BarrierSet&& barrier_set);
    bool Validate(const SyncEnvironment& env, const AccessContext& destination_context, ResourceUsageTag exec_tag) const override;
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag exec_tag) const override;

  private:
    BarrierSet barrier_set_;
};

class SetEventAction : public ReplayAction {
  public:
    SetEventAction(std::shared_ptr<const vvl::Event>&& event, const SyncExecScope& src_exec_scope,
                   std::shared_ptr<const AccessContext>&& src_access_context, const Location& loc);
    bool Validate(const SyncEnvironment& env, const AccessContext& destination_context, ResourceUsageTag exec_tag) const override;
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag exec_tag) const override;

  private:
    vvl::Func command_ = vvl::Func::Empty;
    std::shared_ptr<const vvl::Event> event_;
    // Snapshot of the command buffer's access context at set event time
    std::shared_ptr<const AccessContext> recorded_context_;
    SyncExecScope src_exec_scope_;
};

class ResetEventAction : public ReplayAction {
  public:
    ResetEventAction(std::shared_ptr<const vvl::Event>&& event, const SyncExecScope& exec_scope, const Location& loc);
    bool Validate(const SyncEnvironment& env, const AccessContext& destination_context, ResourceUsageTag exec_tag) const override;
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag exec_tag) const override;

  private:
    vvl::Func command_ = vvl::Func::Empty;
    std::shared_ptr<const vvl::Event> event_;
    SyncExecScope exec_scope_;
};

class WaitEventsAction : public ReplayAction {
  public:
    WaitEventsAction(std::vector<std::shared_ptr<const vvl::Event>>&& events, std::vector<BarrierSet>&& barrier_sets,
                     const Location& loc);
    bool Validate(const SyncEnvironment& env, const AccessContext& destination_context, ResourceUsageTag exec_tag) const override;
    void Apply(SyncEnvironment& env, AccessContext& access_context, ResourceUsageTag exec_tag) const override;

  private:
    vvl::Func command_ = vvl::Func::Empty;
    std::vector<std::shared_ptr<const vvl::Event>> events_;
    std::vector<BarrierSet> barrier_sets_;
};

// Render pass commands do not modify synchronization state during replay. They update the current
// contexts used by first-use validation, since each subpass has its own recorded and destination
// contexts. The context bookkeeping is internal to the replay implementation (sync_op.cpp).
enum class ContextUpdate {
    kNone,
    kBeginRenderPass,
    kNextSubpass,
    kEndRenderPass,
};

struct ReplayEntry {
    ResourceUsageTag tag = 0;

    // Most replay entries are only boundaries between ranges of recorded first accesses. Entries
    // that themselves record an access (currently layout transitions) get an additional check at
    // their tag.
    bool first_use_check = false;

    ContextUpdate context_update = ContextUpdate::kNone;
    const RenderPassAccessContext* rp_context = nullptr;  // kBeginRenderPass only

    // Synchronization state update; null for context update entries
    std::unique_ptr<ReplayAction> action;
};

bool ValidateFirstUseHazards(SyncEnvironment& env, AccessContext& destination_context,
                             const CommandBufferContext& recorded_cb_context, ResourceUsageTag base_tag, const Location& cb_loc);

}  // namespace syncval
