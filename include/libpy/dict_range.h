#pragma once

#include <utility>

#include "libpy/detail/python.h"
#include "libpy/exception.h"
#include "libpy/scoped_ref.h"

namespace py {
/** A range which iterates over the (key, value) pairs of a Python dictionary.
 */
class dict_range {
private:
    PyObject* m_map;

    class iterator {
    public:
        using value_type = std::pair<PyObject*, PyObject*>;
        using reference = value_type&;

    private:
        PyObject* m_map;
        Py_ssize_t m_pos;
        value_type m_item;

    public:
        inline iterator() : m_map(nullptr), m_pos(-1), m_item(nullptr, nullptr) {}

        explicit iterator(PyObject* map);

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
    inline explicit dict_range(PyObject* map) : m_map(map) {}
    inline explicit dict_range(const py::scoped_ref<>& map) : dict_range(map.get()) {}

    static dict_range checked(PyObject* map);
    static dict_range checked(const py::scoped_ref<>& map);

    iterator begin() const;
    iterator end() const;
};
}  // namespace py
