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

#include "sync_access_map.h"

namespace syncval {

AccessMapEntry::AccessMapEntry(const AccessMapEntry& other) : state(other.state) {
    if (other.pattern) pattern = std::make_unique<PatternState>(*other.pattern);
}

void AccessMapEntry::SetUniform(const AccessState& value) {
    if (&value != &state) state.Assign(value);
    pattern.reset();
}

void AccessMapEntry::Clip(const AccessRange& range) {
    if (!pattern) return;
    if (pattern->inside_present && pattern->range.Covers(range)) {
        SetUniform(pattern->inside);
    } else if (pattern->outside_present && !pattern->range.Intersects(range)) {
        pattern.reset();
    }
}

bool AccessMapEntry::CanMerge(const AccessMapEntry& other) const {
    return !pattern && !other.pattern && state.next_global_barrier_index == other.state.next_global_barrier_index &&
           state == other.state;
}

void AccessMap::Materialize(AccessRange range) {
    auto pos = LowerBound(range.begin);
    while (pos != end() && pos->first.begin < range.end) {
        if (!pos->second.pattern) {
            ++pos;
            continue;
        }
        pos = syncval::Split(pos, *this, range);
        const AccessRange bounds = pos->first;
        AccessMapEntry entry(pos->second);
        pos = Erase(pos);
        entry.Visit(bounds, [&](const AccessRange& span, const AccessState* state) {
            if (state) Insert(pos, span, *state);
            return false;
        });
    }
}

void AccessMap::Assign(const AccessMap& other) {
    auto temp_copy(other.impl_map_);
    impl_map_.swap(temp_copy);
}

AccessMap::iterator AccessMap::LowerBound(ResourceAddress range_begin) {
    auto it = impl_map_.lower_bound(AccessRange(range_begin, range_begin));
    return it;
}

AccessMap::const_iterator AccessMap::LowerBound(ResourceAddress range_begin) const {
    auto it = impl_map_.lower_bound(AccessRange(range_begin, range_begin));
    return it;
}

AccessMap::iterator AccessMap::Erase(const iterator& pos) {
    assert(pos != end());
    return impl_map_.erase(pos);
}

void AccessMap::Erase(iterator first, iterator last) {
    auto current = first;
    while (current != last) {
        assert(current != end());
        current = impl_map_.erase(current);
    }
}

AccessMap::iterator AccessMap::Merge(iterator first, iterator last) {
    assert(first != last);
    auto merge_last = last;
    --merge_last;
    assert(first != merge_last);

    const AccessRange merged_range(first->first.begin, merge_last->first.end);
    auto node = impl_map_.extract(merge_last);
    Erase(first, last);
    node.key() = merged_range;
    return impl_map_.insert(last, std::move(node));
}

AccessMap::iterator AccessMap::Insert(const_iterator hint, const AccessRange& range, const AccessMapEntry& access_state) {
    assert(range.non_empty());
    bool hint_open;
    const_iterator impl_next = hint;
    if (impl_map_.empty()) {
        hint_open = true;
    } else if (impl_next == impl_map_.cbegin()) {
        hint_open = range.strictly_less(impl_next->first);
    } else if (impl_next == impl_map_.cend()) {
        auto impl_prev = impl_next;
        --impl_prev;
        hint_open = range.strictly_greater(impl_prev->first);
    } else {
        auto impl_prev = impl_next;
        --impl_prev;
        hint_open = range.strictly_greater(impl_prev->first) && range.strictly_less(impl_next->first);
    }

    if (!hint_open) {
        // Hint was unhelpful, fall back to the non-hinted version
        auto plain_insert = Insert(range, access_state);
        return plain_insert.first;
    }

    auto impl_insert = impl_map_.insert(impl_next, {range, access_state});
    return iterator(impl_insert);
}

std::pair<AccessMap::iterator, bool> AccessMap::Insert(const AccessRange& range, const AccessMapEntry& access_state) {
    assert(range.non_empty());

    // Look for range conflicts (and an insertion point, which makes the lower_bound *not* wasted work)
    // we don't have to check upper if just check that lower doesn't intersect (which it would if lower != upper)
    auto lower = LowerBound(range.begin);
    if (lower == end() || !lower->first.intersects(range)) {
        // range is not even partially overlapped, and lower is strictly > than key
        return {impl_map_.emplace_hint(lower, range, access_state), true};
    }
    // We don't replace
    return {lower, false};
}

AccessMap::iterator AccessMap::InfillGap(const_iterator range_lower_bound, const AccessRange& range,
                                         const AccessState& access_state) {
    assert(LowerBound(range.begin) == range_lower_bound);
    assert(range_lower_bound == end() || range.strictly_less(range_lower_bound->first));
    return impl_map_.insert(range_lower_bound, {range, access_state});
}

void AccessMap::InfillGaps(const AccessRange& range, const AccessState& access_state) {
    auto pos = LowerBound(range.begin);
    ResourceAddress begin = range.begin;
    while (begin < range.end) {
        if (pos == end() || begin < pos->first.begin) {
            const ResourceAddress gap_end = std::min(range.end, pos == end() ? range.end : pos->first.begin);
            Insert(pos, {begin, gap_end}, access_state);
            begin = gap_end;
        } else {
            const AccessRange part(begin, std::min(range.end, pos->first.end));
            if (pos->second.HasGaps()) {
                pos = syncval::Split(pos, *this, part);
                auto& entry = pos->second;
                if (entry.pattern && !entry.pattern->outside_present) {
                    entry.state.Assign(access_state);
                    entry.pattern->outside_present = true;
                }
                if (entry.pattern && !entry.pattern->inside_present) {
                    entry.pattern->inside.Assign(access_state);
                    entry.pattern->inside_present = true;
                }
            }
            begin = part.end;
            ++pos;
        }
    }
}
AccessMap::iterator AccessMap::Split(const iterator split_it, const index_type& index) {
    const auto range = split_it->first;

    if (!range.includes(index)) {
        return split_it;  // If we don't have a valid split point, just return the iterator
    }

    AccessRange lower_range(range.begin, index);

    if (lower_range.empty()) {
        // This is a noop, we're keeping the upper half which is the same as split_it
        return split_it;
    }

    // Save the contents and erase
    auto value = split_it->second;
    auto next_it = impl_map_.erase(split_it);

    AccessRange upper_range(index, range.end);
    assert(!upper_range.empty());  // Upper range cannot be empty

    // Copy value to the upper range
    // NOTE: we insert from upper to lower because that's what emplace_hint can do in constant time
    assert(impl_map_.find(upper_range) == impl_map_.end());
    next_it = impl_map_.emplace_hint(next_it, std::make_pair(upper_range, value));
    next_it->second.Clip(upper_range);

    // Move value to the lower range (we can move since the upper range already got a copy of value)
    assert(impl_map_.find(lower_range) == impl_map_.end());
    next_it = impl_map_.emplace_hint(next_it, std::make_pair(lower_range, std::move(value)));
    next_it->second.Clip(lower_range);

    // Iterator to the beginning of the lower range
    return next_it;
}

AccessMap::iterator Split(AccessMap::iterator in, AccessMap& map, const AccessRange& range) {
    assert(in != map.end());  // Not designed for use with invalid iterators...
    const AccessRange in_range = in->first;
    const AccessRange split_range = in_range & range;

    if (split_range.empty()) {
        return map.end();
    }
    auto pos = in;
    if (split_range.begin != in_range.begin) {
        pos = map.Split(pos, split_range.begin);
        ++pos;
    }
    if (split_range.end != in_range.end) {
        pos = map.Split(pos, split_range.end);
    }
    return pos;
}

void Consolidate(AccessMap& map) {
    using It = AccessMap::iterator;

    It current = map.begin();
    const It map_end = map.end();

    // To be included in a merge range there must be no gap in the AccessRange space, and the mapped_type values must match
    auto can_merge = [](const It& last, const It& cur) {
        return cur->first.begin == last->first.end && cur->second.CanMerge(last->second);
    };

    while (current != map_end) {
        // Establish a trival merge range at the current location, advancing current. Merge range is inclusive of merge_last
        const It merge_first = current;
        It merge_last = current;
        ++current;

        // Expand the merge range as much as possible
        while (current != map_end && can_merge(merge_last, current)) {
            merge_last = current;
            ++current;
        }

        // Current isn't in the active merge range. If there is a non-trivial merge range, we resolve it here.
        if (merge_first != merge_last) {
            map.Merge(merge_first, current);
        }
    }
}

}  // namespace syncval
