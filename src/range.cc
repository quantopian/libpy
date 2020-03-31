#include "libpy/range.h"

namespace py {
range::iterator::iterator(py::borrowed_ref<> it) : m_iterator(it), m_value(nullptr) {
    ++(*this);
}

range::iterator::reference range::iterator::operator*() {
    return m_value;
}

range::iterator::value_type* range::iterator::operator->() {
    return &m_value;
}

range::iterator& range::iterator::operator++() {
    m_value = py::owned_ref(PyIter_Next(m_iterator.get()));
    if (!m_value) {
        if (PyErr_Occurred()) {
            throw py::exception{};
        }
        m_iterator = nullptr;
    }
    return *this;
}

range::iterator range::iterator::operator++(int) {
    range::iterator out = *this;
    return ++out;
}

bool range::iterator::operator!=(const iterator& other) const {
    return !(*this == other);
}

bool range::iterator::operator==(const iterator& other) const {
    return m_iterator == other.m_iterator && m_value.get() == other.m_value.get();
}

range::range(py::borrowed_ref<> iterable) : m_iterator(PyObject_GetIter(iterable.get())) {
    if (!m_iterator) {
        throw py::exception{};
    }
}

range::iterator range::begin() const {
    return range::iterator{m_iterator.get()};
}

range::iterator range::end() const {
    return range::iterator{};
}
}  // namespace py
