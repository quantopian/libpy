#pragma once

#include <utility>

#include "libpy/borrowed_ref.h"
#include "libpy/detail/api.h"
#include "libpy/detail/python.h"
#include "libpy/exception.h"

namespace py {
LIBPY_BEGIN_EXPORT
/** A range which iterates over the (key, value) pairs of a Python dictionary.
 */
class dict_range {
private:
    py::owned_ref<> m_map;

    class iterator {
    public:
        using value_type = std::pair<py::borrowed_ref<>, py::borrowed_ref<>>;
        using reference = value_type&;

    private:
        py::borrowed_ref<> m_map;
        Py_ssize_t m_pos;
        value_type m_item;

    public:
        inline iterator() : m_map(nullptr), m_pos(-1), m_item(nullptr, nullptr) {}

        explicit iterator(py::borrowed_ref<> map);

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
    /** Create an object which iterates a Python dictionary as key, value pairs.

        @note This does not do a type check, `map` must be a Python dictionary.
        @param map The map to create a range over.
    */
    explicit dict_range(py::borrowed_ref<> map);

    /** Assert that `map` is a Python dictionary and then construct a
        `dict_range`.

        @param map The object to check and then make a view over.
        @return A new dict range.
     */
    static dict_range checked(py::borrowed_ref<> map);

    iterator begin() const;
    iterator end() const;
};
LIBPY_END_EXPORT
}  // namespace py
