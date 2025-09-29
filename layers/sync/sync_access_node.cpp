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

#include "sync_access_node.h"
#include "utils/hash_util.h"

size_t ReadNode::Hash() const {
    hash_util::HashCombiner hc;
    hc << stage;
    hc << access_index;
    hc << barriers;
    hc << sync_stages;
    return hc.Value();
}

bool ReadNode::operator==(const ReadNode &other) const {
    return stage == other.stage && access_index == other.access_index && barriers == other.barriers &&
           sync_stages == other.sync_stages;
}

size_t WriteNode::Hash() const {
    hash_util::HashCombiner hc;
    hc << access_index;
    hc << flags;
    hc << barriers.words[0];
    hc << barriers.words[1];
    hc << barriers.words[2];
    hc << dependency_chain;
    return hc.Value();
}

bool WriteNode::operator==(const WriteNode &other) const {
    return access_index == other.access_index && flags == other.flags && barriers == other.barriers &&
           dependency_chain == other.dependency_chain;
}

template <typename NodeType>
NodeType* AccessNodeRegistry::Storage<NodeType>::AllocateNode(const NodeType& def) {
    // Allocate new page if needed
    if (current_page_node_count == kPageSize) {
        pages.emplace_back(Page{std::make_unique<NodeType[]>(kPageSize)});
        current_page_node_count = 0;
    }

    // Allocate and initialize new node
    NodeType* new_node = &pages.back().nodes.get()[current_page_node_count++];
    *new_node = def;

    // Update query map
    node_map.insert(std::make_pair(def, new_node));
    return new_node;
}

template <typename NodeType>
NodeType* AccessNodeRegistry::Storage<NodeType>::GetNode(const NodeType& def) {
    if (auto it = node_map.find(def); it != node_map.end()) {
        return it->second;
    } else {
        return AllocateNode(def);
    }
}

ReadNode* AccessNodeRegistry::GetReadNode(const ReadNode& def) { return read_node_storage_.GetNode(def); }

WriteNode* AccessNodeRegistry::GetWriteNode(const WriteNode& def) { return write_node_storage_.GetNode(def); }

AccessNodeRegistry& GetAccessNodeRegistry() {
    static thread_local AccessNodeRegistry access_node_registry;
    return access_node_registry;
}
