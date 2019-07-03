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
        iterator() : m_map(nullptr), m_pos(-1), m_item(nullptr, nullptr) {}

        explicit iterator(PyObject* map) : m_map(map), m_pos(0) {
            ++(*this);
        }
        iterator(const iterator&) = default;
        iterator& operator=(const iterator&) = default;

        reference operator*() {
            return m_item;
        }

        value_type* operator->() {
            return &m_item;
        }

        iterator& operator++() {
            if (!PyDict_Next(m_map, &m_pos, &m_item.first, &m_item.second)) {
                m_map = nullptr;
                m_pos = -1;
                m_item.first = nullptr;
                m_item.second = nullptr;
            }
            return *this;
        }

        iterator operator++(int) {
            iterator out = *this;
            return ++out;
        }

        bool operator!=(const iterator& other) const {
            return m_pos != other.m_pos;
        }

        bool operator==(const iterator& other) const {
            return m_pos == other.m_pos;
        }
    };

public:
    explicit dict_range(PyObject* map) : m_map(map) {}
    explicit dict_range(const py::scoped_ref<>& map) : dict_range(map.get()) {}

    static dict_range checked(PyObject* map) {
        if (!PyDict_Check(map)) {
            throw py::exception(PyExc_TypeError,
                                "argument to py::dict_range::checked isn't a dict, got: ",
                                Py_TYPE(map)->tp_name);
        }
        return dict_range(map);
    }

    static dict_range checked(const py::scoped_ref<>& map) {
        return checked(map.get());
    }

    iterator begin() const {
        return iterator{m_map};
    }

    iterator end() const {
        return iterator{};
    }
};
}  // namespace py
