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

#include <memory>
#include <vector>
#include "containers/small_vector.h"
#include "utils/math_utils.h"

namespace syncval {

class MemoryPool {
  public:
    static constexpr size_t kPageSize = 256 * 1024;  // bytes

  public:
    // Allocates memory block with a given size and alignment
    void* Allocate(size_t size, size_t alignment);

    // Allocates memory block to store a given number of objects of type T.
    // This only allocates memory. The objects are not initialized/constructed.
    template <typename T>
    void* Allocate(size_t element_count) {
        return Allocate(sizeof(T) * element_count, alignof(T));
    }

    void Reset();

private:
    void AddPage();

  private:
    struct Page {
        std::unique_ptr<uint8_t[]> data;
    };
    std::vector<Page> pages_;

    size_t current_page_ = 0;
    size_t current_page_allocated_ = 0;
};

template <typename T, size_t kObjectsPerPage>
class PagedVector {
    static_assert(IsPowerOfTwo(kObjectsPerPage));
    static constexpr size_t kPageMask = kObjectsPerPage - 1;
    static constexpr size_t kPageShift = MostSignificantBitCompileTime<kObjectsPerPage>();

  public:
    PagedVector(MemoryPool& pool) : pool_(pool) {}

    struct Page {
        T* objects;
        size_t size;

        T* begin() { return objects; }
        const T* begin() const { return objects; }
        T* end() { return objects + size; }
        const T* end() const { return objects + size; }
    };

    Page* begin() { return pages_.begin(); }
    const Page* begin() const { return pages_.begin(); }
    Page* end() { return pages_.end(); }
    const Page* end() const { return pages_.end(); }

    template <typename... Args>
    T& EmplaceBack(Args&&... args) {
        if (pages_.empty() || pages_.back().size == kObjectsPerPage) {
            void* page_ptr = pool_.Allocate<T>(kObjectsPerPage);
            pages_.emplace_back(Page{static_cast<T*>(page_ptr)});
        }
        Page& page = pages_.back();
        T* object_ptr = &page.objects[page.size];
        new (object_ptr) T(args...);
        page.size++;
        return *object_ptr;
    }

    size_t Size() const { return pages_.empty() ? 0 : ((pages_.size() - 1) * kObjectsPerPage + pages_.back().size); }

    T& operator[](size_t index) {
        const uint32_t page_index = uint32_t(index >> kPageShift);
        Page& page = pages_[page_index];
        return page.objects[index & kPageMask];
    }

    const T& operator[](size_t index) const {
        const uint32_t page_index = uint32_t(index >> kPageShift);
        Page& page = pages_[page_index];
        return page.objects[index & kPageMask];
    }

  private:
    MemoryPool& pool_;

    // The list of pages. Each page is a pointer to a memory block allocated from a memory pool.
    // The size of page's memory block is kElementsPerPage * sizeof(T).
    small_vector<Page, 1> pages_;
};

}  // namespace syncval
