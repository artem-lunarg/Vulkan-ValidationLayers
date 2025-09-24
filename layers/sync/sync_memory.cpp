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

#include "sync_memory.h"
#include "utils/math_utils.h"

constexpr size_t kMaxPagesAfterReset = 10;

namespace syncval {

void* MemoryPool::Allocate(size_t size, size_t alignment) {
    assert(size <= kPageSize);

    if (pages_.empty()) {
        AddPage();
    }

    // Adjust current page location based on alignment
    current_page_allocated_ = Align(current_page_allocated_, alignment);

    // Check if there is enough space. If not, grab the next free page or allocate a new one.
    const bool enough_space = (current_page_allocated_ + size <= kPageSize);
    if (!enough_space) {
        current_page_++;
        if (current_page_ == pages_.size()) {
            AddPage();
        }
        current_page_allocated_ = 0;
    }

    void* ptr = &pages_[current_page_].data.get()[current_page_allocated_];
    current_page_allocated_ += size;
    return ptr;
}

void MemoryPool::Reset() {
    pages_.resize(std::min(pages_.size(), kMaxPagesAfterReset));
    current_page_ = 0;
    current_page_allocated_ = 0;
}

void MemoryPool::AddPage() { pages_.emplace_back(Page{std::make_unique<uint8_t[]>(kPageSize)}); }

}  // namespace syncval
