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

#include "sync/sync_common.h"

struct SyncFlag {
    enum : uint32_t {
        kLoadOp = 0x01,
        kStoreOp = 0x02,
        kPresent = 0x04,
        kMarker = 0x08,
    };
};
using SyncFlags = uint32_t;

struct ReadNode {
    VkPipelineStageFlagBits2 stage;
    SyncAccessIndex access_index;
    VkPipelineStageFlags2 barriers;
    VkPipelineStageFlags2 sync_stages;

    size_t Hash() const;
    bool operator==(const ReadNode& other) const;
};

struct WriteNode {
    SyncAccessIndex access_index;
    SyncFlags flags;
    SyncAccessFlags barriers;
    VkPipelineStageFlags2 dependency_chain;

    size_t Hash() const;
    bool operator==(const WriteNode& other) const;
};

class AccessNodeRegistry {
  public:
    ReadNode* GetReadNode(const ReadNode& def);
    WriteNode* GetWriteNode(const WriteNode& def);

  private:
    static constexpr uint32_t kPageSize = 1024;

    template <typename NodeType>
    struct Storage {
        struct Page {
            std::unique_ptr<NodeType[]> nodes;
        };
        struct NodeHasher {
            size_t operator()(const NodeType& node) const { return node.Hash(); }
        };

        NodeType* AllocateNode(const NodeType& def);
        NodeType* GetNode(const NodeType& def);

        std::vector<Page> pages;
        uint32_t current_page_node_count = kPageSize;  // this initial value will add new page upon first access
        vvl::unordered_map<NodeType, NodeType*, NodeHasher> node_map;
    };
    Storage<ReadNode> read_node_storage_;
    Storage<WriteNode> write_node_storage_;
};

AccessNodeRegistry& GetAccessNodeRegistry();