#include "libpy/object_map_key.h"

namespace py {
bool object_map_key::operator==(py::borrowed_ref<> other) const {
    if (!m_ob) {
        return !static_cast<bool>(other);
    }
    if (!other) {
        return false;
    }

    int r = PyObject_RichCompareBool(m_ob.get(), other.get(), Py_EQ);
    if (r < 0) {
        throw py::exception{};
    }

    return r;
}

bool object_map_key::operator!=(py::borrowed_ref<> other) const {
    if (!m_ob) {
        return static_cast<bool>(other);
    }
    if (!other) {
        return true;
    }

    int r = PyObject_RichCompareBool(m_ob.get(), other.get(), Py_NE);
    if (r < 0) {
        throw py::exception{};
    }

    return r;
}

bool object_map_key::operator<(py::borrowed_ref<> other) const {
    if (!m_ob) {
        return false;
    }
    if (!other) {
        return true;
    }

    int r = PyObject_RichCompareBool(m_ob.get(), other.get(), Py_LT);
    if (r < 0) {
        throw py::exception{};
    }

    return r;
}

bool object_map_key::operator<=(py::borrowed_ref<> other) const {
    if (!m_ob) {
        return !static_cast<bool>(other);
    }
    if (!other) {
        return true;
    }

    int r = PyObject_RichCompareBool(m_ob.get(), other.get(), Py_LE);
    if (r < 0) {
        throw py::exception{};
    }

    return r;
}

bool object_map_key::operator>(py::borrowed_ref<> other) const {
    if (!m_ob) {
        return static_cast<bool>(other);
    }
    if (!other) {
        return false;
    }

    int r = PyObject_RichCompareBool(m_ob.get(), other.get(), Py_GT);
    if (r < 0) {
        throw py::exception{};
    }

    return r;
}

bool object_map_key::operator>=(py::borrowed_ref<> other) const {
    if (!m_ob) {
        return true;
    }
    if (!other) {
        return false;
    }

    int r = PyObject_RichCompareBool(m_ob.get(), other.get(), Py_GE);
    if (r < 0) {
        throw py::exception{};
    }

    return r;
}

}  // namespace py
