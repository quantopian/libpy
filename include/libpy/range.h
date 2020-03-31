#pragma once

#include <utility>

#include "libpy/borrowed_ref.h"
#include "libpy/detail/api.h"
#include "libpy/detail/python.h"
#include "libpy/exception.h"
#include "libpy/owned_ref.h"

namespace py {
LIBPY_BEGIN_EXPORT
/** A range which lazily iterates over the elements of a Python iterable.
 */
class range {
private:
    py::owned_ref<> m_iterator;

    class iterator {
    public:
        using value_type = py::owned_ref<>;
        using reference = value_type&;

    private:
        py::borrowed_ref<> m_iterator;
        value_type m_value;

    public:
        inline iterator() : m_iterator(nullptr), m_value(nullptr) {}

        explicit iterator(py::borrowed_ref<> it);

        iterator(const iterator&) = default;
        iterator& operator=(const iterator&) = default;

        reference operator*();
        value_type* operator->();

        iterator& operator++();
        iterator operator++(int);

        bool operator!=(const iterator& other) const;
        bool operator==(const iterator& other) const;
    };

public:
    explicit range(py::borrowed_ref<> iterable);

    iterator begin() const;
    iterator end() const;
};
LIBPY_END_EXPORT
}  // namespace py
