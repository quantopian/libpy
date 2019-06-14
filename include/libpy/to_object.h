#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Python.h>

#include "libpy/scoped_ref.h"

namespace py {
namespace dispatch {
/** Dispatch struct for defining overrides for `py::to_object`.

    To add a new dispatch, add an explicit template specialization for the type
    to convert with a static member function `f` which accepts the dispatched
    type and returns a `PyObject*`.
 */
template<typename T>
struct to_object;
}  // namespace dispatch

namespace detail {
template<typename T>
struct has_to_object {
private:
    template<typename U>
    static decltype(py::dispatch::to_object<U>::f(std::declval<U>()), std::true_type{})
    test(int);

    template<typename>
    static std::false_type test(long);

public:
    static constexpr bool value = std::is_same_v<decltype(test<T>(0)), std::true_type>;
};
}  // namespace detail

/** Compile time boolean to detect if `to_object` works for a given type. This exists to
    make it easier to use `if constexpr` to test this condition instead of using more
    complicated SFINAE.
 */
template<typename T>
constexpr bool has_to_object = detail::has_to_object<T>::value;

/** Convert a C++ object into a Python object recursively.

    @param ob The object to convert
    @return `ob` as a Python object.
    @see py::dispatch::to_object
 */
template<typename T>
scoped_ref<> to_object(T&& ob) {
    using underlying_type = std::remove_cv_t<std::remove_reference_t<T>>;
    return scoped_ref(dispatch::to_object<underlying_type>::f(std::forward<T>(ob)));
}

namespace dispatch {
template<typename C>
PyObject* sequence_to_list(const C& v) {
    py::scoped_ref out(PyList_New(v.size()));
    if (!out) {
        return nullptr;
    }

    std::size_t ix = 0;
    for (const auto& elem : v) {
        PyObject* ob = py::to_object(elem).escape();
        if (!ob) {
            return nullptr;
        }

        PyList_SET_ITEM(out.get(), ix++, ob);
    }

    return std::move(out).escape();
}

template<std::size_t n>
struct to_object<std::array<char, n>> {
    static PyObject* f(const std::array<char, n>& cs) {
        return PyBytes_FromStringAndSize(cs.data(), n);
    }
};

template<typename T, std::size_t n>
struct to_object<std::array<T, n>> {
    static PyObject* f(const std::array<T, n>& v) {
        return sequence_to_list(v);
    }
};

/** Identity conversion for `scoped_ref`.
 */
template<typename T>
struct to_object<scoped_ref<T>> {
    static PyObject* f(scoped_ref<T> ob) {
        return reinterpret_cast<PyObject*>(std::move(ob).escape());
    }
};

template<>
struct to_object<std::string> {
    static PyObject* f(const std::string& cs) {
        return PyBytes_FromStringAndSize(cs.data(), cs.size());
    }
};

template<>
struct to_object<std::string_view> {
    static PyObject* f(const std::string_view& cs) {
        return PyBytes_FromStringAndSize(cs.data(), cs.size());
    }
};

template<std::size_t n>
struct to_object<char[n]> {
    static PyObject* f(const char* cs) {
        return PyBytes_FromStringAndSize(cs, n - 1);
    }
};

template<>
struct to_object<bool> {
    static PyObject* f(bool value) {
        return PyBool_FromLong(value);
    }
};

namespace detail {
template<typename T>
struct int_to_object {
    static PyObject* f(T value) {

        // convert the object to the widest type for the given signedness
        if constexpr (std::is_signed_v<T>) {
            return PyLong_FromLongLong(value);
        }
        else {
            return PyLong_FromUnsignedLongLong(value);
        }
    }
};
}  // namespace detail

template<>
struct to_object<signed long long> : public detail::int_to_object<signed long long> {};

template<>
struct to_object<signed long> : public detail::int_to_object<signed long> {};

template<>
struct to_object<signed int> : public detail::int_to_object<signed int> {};

template<>
struct to_object<signed short> : public detail::int_to_object<signed short> {};

template<>
struct to_object<signed char> : public detail::int_to_object<signed char> {};

template<>
struct to_object<unsigned long long> : public detail::int_to_object<unsigned long long> {
};

template<>
struct to_object<unsigned long> : public detail::int_to_object<unsigned long> {};

template<>
struct to_object<unsigned int> : public detail::int_to_object<unsigned int> {};

template<>
struct to_object<unsigned short> : public detail::int_to_object<unsigned short> {};

template<>
struct to_object<unsigned char> : public detail::int_to_object<unsigned char> {};

template<>
struct to_object<float> {
    static PyObject* f(float value) {
        return PyFloat_FromDouble(value);
    }
};

template<>
struct to_object<double> {
    static PyObject* f(double value) {
        return PyFloat_FromDouble(value);
    }
};

template<typename M>
struct map_to_object {
    static PyObject* f(const M& m) {
        py::scoped_ref out(PyDict_New());

        if (!out) {
            return nullptr;
        }

        for (const auto& [key, value] : m) {
            auto key_ob = py::to_object(key);
            if (!key_ob) {
                return nullptr;
            }
            auto value_ob = py::to_object(value);
            if (!value_ob) {
                return nullptr;
            }

            if (PyDict_SetItem(out.get(), key_ob.get(), value_ob.get())) {
                return nullptr;
            }
        }

        return std::move(out).escape();
    }
};

template<typename K, typename V, typename Hash, typename KeyEqual>
struct to_object<std::unordered_map<K, V, Hash, KeyEqual>>
    : public map_to_object<std::unordered_map<K, V, Hash, KeyEqual>> {};

template<typename T>
struct to_object<std::vector<T>> {
    static PyObject* f(const std::vector<T>& v) {
        return sequence_to_list(v);
    }
};

template<typename S>
struct set_to_object {
    static PyObject* f(const S& s) {
        py::scoped_ref out(PySet_New(nullptr));

        if (!out) {
            return nullptr;
        }

        for (const auto& elem : s) {
            auto ob = py::to_object(elem);
            if (!ob) {
                return nullptr;
            }
            if (PySet_Add(out.get(), ob.get())) {
                return nullptr;
            }
        }

        return std::move(out).escape();
    }
};

template<typename T>
struct to_object<std::unordered_set<T>> : set_to_object<std::unordered_set<T>> {};

template<typename... Ts>
struct to_object<std::tuple<Ts...>> {
private:
    template<typename T, std::size_t... Ix>
    static bool
    fill_tuple_as_objects(PyObject* out, T&& tup, std::index_sequence<Ix...>) {
        bool result = false;
        auto f = [out, &result](std::size_t ix, const auto& elem) {
            PyObject* as_object = py::to_object(elem).escape();
            if (!as_object) {
                result = true;
            }
            else {
                PyTuple_SET_ITEM(out, ix, as_object);
            }
            return '\0';
        };
        (..., f(Ix, std::get<Ix>(tup)));
        return result;
    }

public:
    static PyObject* f(const std::tuple<Ts...>& tup) {
        py::scoped_ref out(PyTuple_New(sizeof...(Ts)));

        if (!out) {
            return nullptr;
        }

        if (fill_tuple_as_objects(out.get(), tup, std::index_sequence_for<Ts...>{})) {
            return nullptr;
        }

        return std::move(out).escape();
    }
};

template<typename T, typename U>
struct to_object<std::pair<T, U>> {
public:
    static PyObject* f(const std::pair<T, U>& p) {
        // Delegate to std::tuple dispatch.
        return std::move(py::to_object(std::make_tuple(std::get<0>(p), std::get<1>(p))))
            .escape();
    }
};

template<typename T>
struct to_object<std::optional<T>> {
    static PyObject* f(const std::optional<T>& maybe_value) {
        if (!maybe_value) {
            Py_RETURN_NONE;
        }

        return py::to_object(*maybe_value);
    }
};
}  // namespace dispatch
}  // namespace py
