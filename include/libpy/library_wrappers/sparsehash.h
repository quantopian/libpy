#pragma once

#include <stdexcept>

#include <sparsehash/dense_hash_map>
#include <sparsehash/dense_hash_set>
#include <sparsehash/sparse_hash_map>
#include <sparsehash/sparse_hash_set>

#include "libpy/to_object.h"

namespace py {
/** A wrapper around `google::dense_hash_map` which uses `std::hash` instead of
    `tr1::hash` and requires an empty key at construction time.

    @tparam Key The key type.
    @tparam T The value type.
    @tparam HashFcn The key hash functor type.
    @tparam Alloc The allocator object to use. In general, don't change this.
 */
template<typename Key,
         typename T,
         typename HashFcn = std::hash<Key>,  // change the default to std::hash
         typename EqualKey = std::equal_to<Key>,
         typename Alloc = google::libc_allocator_with_realloc<std::pair<const Key, T>>>
struct dense_hash_map : public google::dense_hash_map<Key, T, HashFcn, EqualKey, Alloc> {
private:
    using base = google::dense_hash_map<Key, T, HashFcn, EqualKey, Alloc>;

public:
    dense_hash_map() = delete;  // User must give a missing value.

    /**
       @param empty_key An element of type `Key` which denotes an empty slot.
                        This value can not itself be used as a valid key.
       @param expected_size A size hint for the map.
     */
    dense_hash_map(const Key& empty_key, std::size_t expected_size = 0)
        : base(expected_size) {
        if (empty_key != empty_key) {
            // the first insert will hang forever if `empty_key != empty_key`
            throw std::invalid_argument{"dense_hash_map: empty_key != empty_key"};
        }
        this->set_empty_key(empty_key);
    }

    dense_hash_map(const dense_hash_map& cpfrom) : base(cpfrom) {}

    dense_hash_map(dense_hash_map&& mvfrom) noexcept {
        this->swap(mvfrom);
    }

    dense_hash_map& operator=(const dense_hash_map& cpfrom) {
        base::operator=(cpfrom);
        return *this;
    }

    dense_hash_map& operator=(dense_hash_map&& mvfrom) noexcept {
        this->swap(mvfrom);
        return *this;
    }
};

/** A wrapper around `google::sparse_hash_map` which uses `std::hash` instead of
    `tr1::hash` and requires an empty key at construction time.

    @tparam Key The key type.
    @tparam T The value type.
    @tparam HashFcn The key hash functor type.
    @tparam Alloc The allocator object to use. In general, don't change this.
 */
template<typename Key,
         typename T,
         typename HashFcn = std::hash<Key>,  // change the default to std::hash
         typename EqualKey = std::equal_to<Key>,
         typename Alloc = google::libc_allocator_with_realloc<std::pair<const Key, T>>>
struct sparse_hash_map
    : public google::sparse_hash_map<Key, T, HashFcn, EqualKey, Alloc> {
private:
    using base = google::sparse_hash_map<Key, T, HashFcn, EqualKey, Alloc>;

public:
    using base::sparse_hash_map;

    sparse_hash_map(const sparse_hash_map& cpfrom) : base(cpfrom) {}

    sparse_hash_map(sparse_hash_map&& mvfrom) noexcept {
        this->swap(mvfrom);
    }

    sparse_hash_map& operator=(const sparse_hash_map& cpfrom) {
        base::operator=(cpfrom);
        return *this;
    }

    sparse_hash_map& operator=(sparse_hash_map&& mvfrom) noexcept {
        this->swap(mvfrom);
        return *this;
    }
};

template<typename Key,
         typename HashFcn = std::hash<Key>,
         typename EqualKey = std::equal_to<Key>,
         typename Alloc = google::libc_allocator_with_realloc<Key>>
struct dense_hash_set : public google::dense_hash_set<Key, HashFcn, EqualKey, Alloc> {
private:
    using base = google::dense_hash_set<Key, HashFcn, EqualKey, Alloc>;

public:
    dense_hash_set() = delete;  // User must give a missing value.

    /**
       @param empty_key An element of type `Key` which denotes an empty slot.
                        This value can not itself be used as a valid key.
       @param expected_size A size hint for the set.
     */
    dense_hash_set(const Key& empty_key, std::size_t expected_size = 0)
        : base(expected_size) {
        if (empty_key != empty_key) {
            // the first insert will hang forever if `empty_key != empty_key`
            throw std::invalid_argument{"dense_hash_set: empty_key != empty_key"};
        }
        this->set_empty_key(empty_key);
    }

    dense_hash_set(const dense_hash_set& cpfrom) : base(cpfrom) {}

    dense_hash_set(dense_hash_set&& mvfrom) noexcept {
        this->swap(mvfrom);
    }

    dense_hash_set& operator=(const dense_hash_set& cpfrom) {
        base::operator=(cpfrom);
        return *this;
    }

    dense_hash_set& operator=(dense_hash_set&& mvfrom) noexcept {
        this->swap(mvfrom);
        return *this;
    }
};

namespace dispatch {
template<typename Key, typename T, typename HashFcn, typename EqualKey, typename Alloc>
struct to_object<dense_hash_map<Key, T, HashFcn, EqualKey, Alloc>>
    : public map_to_object<dense_hash_map<Key, T, HashFcn, EqualKey, Alloc>> {};

template<typename Key, typename T, typename HashFcn, typename EqualKey, typename Alloc>
struct to_object<sparse_hash_map<Key, T, HashFcn, EqualKey, Alloc>>
    : public map_to_object<sparse_hash_map<Key, T, HashFcn, EqualKey, Alloc>> {};

template<typename Key, typename HashFcn, typename EqualKey, typename Alloc>
struct to_object<dense_hash_set<Key, HashFcn, EqualKey, Alloc>>
    : public set_to_object<dense_hash_set<Key, HashFcn, EqualKey, Alloc>> {};

template<typename Key, typename T, typename HashFcn, typename EqualKey, typename Alloc>
struct to_object<google::dense_hash_map<Key, T, HashFcn, EqualKey, Alloc>>
    : public map_to_object<google::dense_hash_map<Key, T, HashFcn, EqualKey, Alloc>> {};

template<typename Key, typename T, typename HashFcn, typename EqualKey, typename Alloc>
struct to_object<google::sparse_hash_map<Key, T, HashFcn, EqualKey, Alloc>>
    : public map_to_object<google::sparse_hash_map<Key, T, HashFcn, EqualKey, Alloc>> {};

template<typename Key, typename HashFcn, typename EqualKey, typename Alloc>
struct to_object<google::dense_hash_set<Key, HashFcn, EqualKey, Alloc>>
    : public set_to_object<google::dense_hash_set<Key, HashFcn, EqualKey, Alloc>> {};

template<typename Key, typename HashFcn, typename EqualKey, typename Alloc>
struct to_object<google::sparse_hash_set<Key, HashFcn, EqualKey, Alloc>>
    : public set_to_object<google::sparse_hash_set<Key, HashFcn, EqualKey, Alloc>> {};

}  // namespace dispatch
}  // namespace py
