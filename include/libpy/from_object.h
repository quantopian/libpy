#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Python.h>

#include "libpy/demangle.h"
#include "libpy/exception.h"
#include "libpy/scoped_ref.h"

namespace py {
namespace dispatch {
/** Dispatch struct for defining overrides for `py::from_object`.

    To add a new dispatch, add an explicit template specialization for the type
    to convert with a static member function `f` which accepts `PyObject*` and returns
    a new object of type `T`.
 */
template<typename T>
struct from_object;
}  // namespace dispatch

/** Exception raised when an invalid `py::from_object` call is performed.
 */
class invalid_conversion : public std::exception {
private:
    std::string m_msg;

    inline invalid_conversion(const std::string& msg) : m_msg(msg) {}

public:
    template<typename ConvertTo>
    static invalid_conversion make(PyObject* ob) {
        auto repr = scoped_ref(PyObject_Repr(ob));
        if (!repr) {
            throw py::exception("failed to call repr on ob");
        }
        const char* data = PyUnicode_AsUTF8(repr.get());
        if (!data) {
            throw py::exception("failed to get utf8 string from repr result");
        }

        std::stringstream s;
        s << "failed to convert Python object of type " << Py_TYPE(ob)->tp_name
          << " to a C++ object of type " << py::util::type_name<ConvertTo>().get()
          << ": ob=" << data;
        return invalid_conversion(s.str());
    }

    inline const char* what() const noexcept {
        return m_msg.data();
    }
};

/** Convert a C++ object into a Python object recursively.

    @param ob The object to convert
    @return `ob` as a Python object.
    @see py::dispatch::from_object
 */
template<typename T>
T from_object(PyObject* ob) {
    return dispatch::from_object<T>::f(ob);
}

/** Convert a Python object into a C++ object recursively.


    @param ob The object to convert
    @return `ob` as a C++ object.
    @see py::dispatch::from_object
 */
template<typename T, typename U>
T from_object(scoped_ref<U>& ob) {
    return dispatch::from_object<T>::f(static_cast<PyObject*>(ob));
}

namespace dispatch {
template<std::size_t n>
struct from_object<std::array<char, n>> {
    static std::array<char, n> f(PyObject* cs) {
        std::array<char, n> out = {0};

        const char* data;
        Py_ssize_t size;

        if (PyBytes_Check(cs)) {
            size = PyBytes_GET_SIZE(cs);
            data = PyBytes_AS_STRING(cs);
        }
        else if (PyUnicode_Check(cs)) {
            if (!(data = PyUnicode_AsUTF8AndSize(cs, &size))) {
                throw py::exception("failed to convert unicode to string");
            }
        }
        else {
            throw invalid_conversion::make<std::array<char, n>>(cs);
        }

        std::copy_n(data, std::min(n, static_cast<std::size_t>(size)), out.begin());
        return out;
    }
};

template<>
struct from_object<char> {
    static char f(PyObject* cs) {
        const char* data;
        Py_ssize_t size;

        if (PyBytes_Check(cs)) {
            size = PyBytes_GET_SIZE(cs);
            data = PyBytes_AS_STRING(cs);
        }
        else if (PyUnicode_Check(cs)) {
            if (!(data = PyUnicode_AsUTF8AndSize(cs, &size))) {
                throw py::exception("failed to convert unicode to string");
            }
        }
        else {
            throw invalid_conversion::make<char>(cs);
        }

        if (size != 1) {
            throw invalid_conversion::make<char>(cs);
        }

        return data[0];
    }
};

/** Identity conversion for `PyObject*`.
 */
template<>
struct from_object<PyObject*> {
    static PyObject* f(PyObject* ob) {
        return ob;
    }
};

/** Identity conversion for `scoped_ref`.
 */
template<typename T>
struct from_object<scoped_ref<T>> {
    static scoped_ref<T> f(PyObject* ob) {
        Py_INCREF(ob);
        return scoped_ref(ob);
    }
};

template<>
struct from_object<std::string_view> {
    static std::string_view f(PyObject* cs) {
        const char* data;
        Py_ssize_t size;

        if (PyBytes_Check(cs)) {
            size = PyBytes_GET_SIZE(cs);
            data = PyBytes_AS_STRING(cs);
        }
        else if (PyUnicode_Check(cs)) {
            if (!(data = PyUnicode_AsUTF8AndSize(cs, &size))) {
                throw py::exception("failed to convert unicode to string");
            }
        }
        else {
            throw invalid_conversion::make<std::string_view>(cs);
        }

        return std::string_view(data, size);
    }
};

template<>
struct from_object<std::string> {
    static std::string f(PyObject* cs) {
        return std::string(py::from_object<std::string_view>(cs));
    }
};

namespace detail {
template<typename T>
struct int_from_object {
private:
    // the widest type for the given signedness
    using wide_type =
        std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>;

    static_assert(sizeof(T) <= sizeof(wide_type),
                  "cannot use int_from_object with a type wider than the wide_type for "
                  "the given signedness");

public:
    static T f(PyObject* value) {
        wide_type wide;

        // convert the object to the widest type for the given signedness
        if constexpr (std::is_signed_v<T>) {
            wide = PyLong_AsLongLong(value);
        }
        else {
            wide = PyLong_AsUnsignedLongLong(value);
        }

        // check if `value` would overflow `wide_type`
        if (PyErr_Occurred()) {
            throw invalid_conversion::make<T>(value);
        }

        // check if narrowing the wide value to the given type would overflow
        if (wide > std::numeric_limits<T>::max() ||
            wide < std::numeric_limits<T>::min()) {
            py::raise(PyExc_OverflowError)
                << "converting " << value << " to type" << py::util::type_name<T>().get()
                << " overflows";
            throw invalid_conversion::make<T>(value);
        }
        return wide;
    }
};
}  // namespace detail

template<>
struct from_object<signed long long> : public detail::int_from_object<signed long long> {
};

template<>
struct from_object<signed long> : public detail::int_from_object<signed long> {};

template<>
struct from_object<signed int> : public detail::int_from_object<signed int> {};

template<>
struct from_object<signed short> : public detail::int_from_object<signed short> {};

template<>
struct from_object<signed char> : public detail::int_from_object<signed char> {};

template<>
struct from_object<unsigned long long>
    : public detail::int_from_object<unsigned long long> {};

template<>
struct from_object<unsigned long> : public detail::int_from_object<unsigned long> {};

template<>
struct from_object<unsigned int> : public detail::int_from_object<unsigned int> {};

template<>
struct from_object<unsigned short> : public detail::int_from_object<unsigned short> {};

template<>
struct from_object<unsigned char> : public detail::int_from_object<unsigned char> {};

template<>
struct from_object<double> {
    static double f(PyObject* value) {
        double out = PyFloat_AsDouble(value);
        if (out == -1.0 && PyErr_Occurred()) {
            throw invalid_conversion::make<double>(value);
        }
        return out;
    }
};

template<typename K, typename V, typename Hash, typename KeyEqual>
struct from_object<std::unordered_map<K, V, Hash, KeyEqual>> {
    using map_type = std::unordered_map<K, V, Hash, KeyEqual>;

    static map_type f(PyObject* m) {
        if (!PyDict_Check(m)) {
            throw invalid_conversion::make<map_type>(m);
        }

        map_type out;

        Py_ssize_t pos = 0;
        PyObject* key;
        PyObject* value;

        while (PyDict_Next(m, &pos, &key, &value)) {
            out.emplace(py::from_object<K>(key), py::from_object<V>(value));
        }

        return out;
    }
};

template<typename T>
struct from_object<std::vector<T>> {
    static std::vector<T> f(PyObject* v) {
        if (!PyList_Check(v)) {
            throw invalid_conversion::make<std::vector<T>>(v);
        }

        std::vector<T> out;

        Py_ssize_t size = PyList_GET_SIZE(v);
        for (Py_ssize_t ix = 0; ix < size; ++ix) {
            out.emplace_back(py::from_object<T>(PyList_GET_ITEM(v, ix)));
        }

        return out;
    }
};

template<typename T>
struct from_object<std::unordered_set<T>> {
    static std::unordered_set<T> f(PyObject* s) {
        if (!(PySet_Check(s) || PyFrozenSet_Check(s))) {
            throw invalid_conversion::make<std::unordered_set<T>>(s);
        }

        std::unordered_set<T> out;

        auto it = scoped_ref(PyObject_GetIter(s));
        if (!it) {
            throw py::exception("failed to make iterator");
        }

        while (auto item = scoped_ref(PyIter_Next(it.get()))) {
            out.emplace(py::from_object<T>(item));
        }
        if (PyErr_Occurred()) {
            throw py::exception("python error occurred while iterating");
        }

        return out;
    }
};

template<typename... Ts>
struct from_object<std::tuple<Ts...>> {
private:
    template<std::size_t... ixs>
    static std::tuple<Ts...> fill_tuple(PyObject* tup, std::index_sequence<ixs...>) {
        return {py::from_object<Ts>(PyTuple_GET_ITEM(tup, ixs))...};
    }

public:
    static std::tuple<Ts...> f(PyObject* tup) {
        if (!(PyTuple_Check(tup) || PyTuple_GET_SIZE(tup) != sizeof...(Ts))) {
            throw invalid_conversion::make<std::tuple<Ts...>>(tup);
        }

        return fill_tuple(tup, std::index_sequence_for<Ts...>{});
    }
};

template<typename T>
struct from_object<std::optional<T>> {
    static std::optional<T> f(PyObject* maybe_value) {
        if (maybe_value == nullptr || maybe_value == Py_None) {
            return std::nullopt;
        }

        return py::from_object<T>(maybe_value);
    }
};
}  // namespace dispatch
}  // namespace py
