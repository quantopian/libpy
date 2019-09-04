#pragma once

#include <utility>

#include "libpy/detail/python.h"
#include "libpy/exception.h"
#include "libpy/scoped_ref.h"

namespace py {
/** A range which lazily iterates over the elements of a Python iterable.
 */
class range {
private:
    py::scoped_ref<> m_iterator;

    class iterator {
    public:
        using value_type = py::scoped_ref<>;
        using reference = value_type&;

    private:
        PyObject* m_iterator;
        value_type m_value;

    public:
        inline iterator() : m_iterator(nullptr), m_value(nullptr) {}

        explicit iterator(PyObject* it);

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
    explicit range(PyObject* iterable);
    inline explicit range(const py::scoped_ref<>& iterable) : range(iterable.get()) {}

    iterator begin() const;
    iterator end() const;
};
}  // namespace py
