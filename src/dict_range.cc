#include "libpy/dict_range.h"

namespace py {
dict_range::dict_range(py::borrowed_ref<> map)
    : m_map(py::owned_ref<>::new_reference(map)) {}

dict_range::iterator::iterator(py::borrowed_ref<> map) : m_map(map), m_pos(0) {
    ++(*this);
}

dict_range::iterator::reference dict_range::iterator::operator*() {
    return m_item;
}

dict_range::iterator::value_type* dict_range::iterator::operator->() {
    return &m_item;
}

dict_range::iterator& dict_range::iterator::operator++() {
    PyObject* k;
    PyObject* v;
    if (!PyDict_Next(m_map.get(), &m_pos, &k, &v)) {
        m_map = nullptr;
        m_pos = -1;
        m_item.first = nullptr;
        m_item.second = nullptr;
    }
    else {
        m_item.first = k;
        m_item.second = v;
    }
    return *this;
}

dict_range::iterator dict_range::iterator::operator++(int) {
    dict_range::iterator out = *this;
    return ++out;
}

bool dict_range::iterator::operator!=(const iterator& other) const {
    return m_pos != other.m_pos;
}

bool dict_range::iterator::operator==(const iterator& other) const {
    return m_pos == other.m_pos;
}

dict_range dict_range::checked(py::borrowed_ref<> map) {
    if (!PyDict_Check(map)) {
        throw py::exception(PyExc_TypeError,
                            "argument to py::dict_range::checked isn't a dict, got: ",
                            Py_TYPE(map)->tp_name);
    }
    return dict_range(map);
}

dict_range::iterator dict_range::begin() const {
    return dict_range::iterator{m_map};
}

dict_range::iterator dict_range::end() const {
    return dict_range::iterator{};
}
}  // namespace py
