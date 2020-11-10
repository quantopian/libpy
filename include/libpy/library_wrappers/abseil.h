#pragma once

#include <absl/container/btree_map.h>
#include <absl/container/btree_set.h>

#include "libpy/to_object.h"

namespace py {
namespace dispatch {

template<typename Key, typename T>
struct to_object<absl::btree_map<Key, T>>
    : public map_to_object<absl::btree_map<Key, T>> {};

template<typename Key>
struct to_object<absl::btree_set<Key>> : public set_to_object<absl::btree_set<Key>> {};

}  // namespace dispatch
}  // namespace py
