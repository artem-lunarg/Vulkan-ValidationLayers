/* Copyright (c) 2025 The Khronos Group Inc.
 * Copyright (c) 2025 Valve Corporation
 * Copyright (c) 2025 LunarG, Inc.
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

// NOTE: SyncAccessFlags struct replaces what was originally defined as std::bitset<192>.
// This is done for simplicity and efficiency purposes. The general nature of std::bitset
// makes it inefficient in some scenarios (e.g. hashing). Implementation simplicity is
// important because SyncAccessFlags is one of the main types in syncval implementation.

struct SyncAccessFlags {
    static constexpr size_t kWordCount = 3;
    static constexpr size_t kBitCount = kWordCount * sizeof(uint64_t) * 8;

    uint64_t words[kWordCount];

    SyncAccessFlags() {
        words[0] = 0;
        words[1] = 0;
        words[2] = 0;
    }

    SyncAccessFlags(uint64_t value) {
        words[0] = value;
        words[1] = 0;
        words[2] = 0;
    }

    SyncAccessFlags(uint64_t word0, uint64_t word1, uint64_t word2) {
        words[0] = word0;
        words[1] = word1;
        words[2] = word2;
    }

    SyncAccessFlags& reset() {
        words[0] = 0;
        words[1] = 0;
        words[2] = 0;
        return *this;
    }

    // NOTE: throught implementation (pos >> 6) or equivalently pos/64 determines the 64-bit word index.
    // Then (pos & 63u) is the bit index within the selected word.

    SyncAccessFlags& reset(size_t pos) {
        assert(pos < kBitCount);
        words[pos >> 6] &= ~(uint64_t(1) << (pos & 63u));
        return *this;
    }

    SyncAccessFlags& set(size_t pos) {
        assert(pos < kBitCount);
        words[pos >> 6] |= (uint64_t(1) << (pos & 63u));
        return *this;
    }

    SyncAccessFlags& operator|=(const SyncAccessFlags& other) {
        words[0] |= other.words[0];
        words[1] |= other.words[1];
        words[2] |= other.words[2];
        return *this;
    }

    SyncAccessFlags operator|(const SyncAccessFlags& other) const {
        return SyncAccessFlags((words[0] | other.words[0]), (words[1] | other.words[1]), (words[2] | other.words[2]));
    }

    SyncAccessFlags operator&(const SyncAccessFlags& other) const {
        return SyncAccessFlags((words[0] & other.words[0]), (words[1] & other.words[1]), (words[2] & other.words[2]));
    }

    SyncAccessFlags operator~() const { return SyncAccessFlags(~words[0], ~words[1], ~words[2]); }

    SyncAccessFlags operator<<(size_t pos) const {
        assert(pos < kBitCount);

        // The assert shows the only use case used by syncval.
        // This general operator will be replaced by specific function and assert won't be needed.
        assert(words[0] == 1 && words[1] == 0 && words[2] == 0);

        uint64_t words[3] = {};
        words[pos >> 6] = (uint64_t(1) << (pos & 63u));
        return SyncAccessFlags(words[0], words[1], words[2]);
    }

    bool operator==(const SyncAccessFlags& other) const {
        return words[0] == other.words[0] && words[1] == other.words[1] && words[2] == other.words[2];
    }

    bool operator[](size_t pos) const {
        assert(pos < kBitCount);
        return (words[pos >> 6] & (uint64_t(1) << (pos & 63u))) != 0;
    }

    bool none() const { return words[0] == 0 && words[1] == 0 && words[2] == 0; }
    bool any() const { return !none(); }
    size_t size() const { return kBitCount; }
};
