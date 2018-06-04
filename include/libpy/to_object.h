#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
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

/** Convert a C++ object into a Python object recursively.

    @param ob The object to convert
    @return `ob` as a Python object.
    @see py::dispatch::to_object
 */
template<typename T>
scoped_ref<PyObject> to_object(T&& ob) {
    using underlying_type = std::remove_cv_t<std::remove_reference_t<T>>;
    return scoped_ref(dispatch::to_object<underlying_type>::f(std::forward<T>(ob)));
}

namespace dispatch {
template<std::size_t n>
struct to_object<std::array<char, n>> {
    static PyObject* f(const std::array<char, n>& cs) {
        return PyUnicode_FromStringAndSize(cs.data(), n);
    }
};

/** Identity conversion for `scoped_ref`.
 */
template<typename T>
struct to_object<scoped_ref<T>> {
    static PyObject* f(scoped_ref<T>& ob) {
        PyObject* out = ob.get();
        Py_XINCREF(out);
        return out;
    }
};

template<>
struct to_object<std::string> {
    static PyObject* f(const std::string& cs) {
        return PyUnicode_FromStringAndSize(cs.data(), cs.size());
    }
};

template<>
struct to_object<std::string_view> {
    static PyObject* f(const std::string_view& cs) {
        return PyUnicode_FromStringAndSize(cs.data(), cs.size());
    }
};

template<std::size_t n>
struct to_object<char[n]> {
    static PyObject* f(const char* cs) {
        return PyUnicode_FromStringAndSize(cs, n - 1);
    }
};

template<>
struct to_object<std::size_t> {
    static PyObject* f(std::size_t value) {
        return PyLong_FromSize_t(value);
    }
};

template<>
struct to_object<std::int64_t> {
    static PyObject* f(std::int64_t value) {
        return PyLong_FromLong(value);
    }
};

template<typename M>
struct map_to_object {
    static PyObject* f(const M& m) {
        auto out = py::scoped_ref(PyDict_New());

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

    static PyObject* f(M& m) {
        auto out = py::scoped_ref(PyDict_New());

        if (!out) {
            return nullptr;
        }

        for (auto& [key, value] : m) {
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
        auto out = py::scoped_ref(PyList_New(v.size()));

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

    static PyObject* f(std::vector<T>& v) {
        auto out = py::scoped_ref(PyList_New(v.size()));

        if (!out) {
            return nullptr;
        }

        std::size_t ix = 0;
        for (auto& elem : v) {
            PyObject* ob = py::to_object(elem).escape();
            if (!ob) {
                return nullptr;
            }

            PyList_SET_ITEM(out.get(), ix++, ob);
        }

        return std::move(out).escape();
    }
};

template<typename T>
struct to_object<std::unordered_set<T>> {
    static PyObject* f(const std::unordered_set<T>& s) {
        auto out = py::scoped_ref(PySet_New(nullptr));

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

    static PyObject* f(std::unordered_set<T>& s) {
        auto out = py::scoped_ref(PySet_New(nullptr));

        if (!out) {
            return nullptr;
        }

        for (auto& elem : s) {
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

template<typename... Ts>
struct to_object<std::tuple<Ts...>> {
private:
    template<typename T, std::size_t... Ix>
    static bool fill_tuple_as_objects(PyObject* out,
                                      T&& tup,
                                      std::index_sequence<Ix...>) {
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
        (... , f(Ix, std::get<Ix>(tup)));
        return result;
    }

public:
    static PyObject* f(const std::tuple<Ts...>& tup) {
        auto out = py::scoped_ref(PyTuple_New(sizeof...(Ts)));

        if (!out) {
            return nullptr;
        }

        if (fill_tuple_as_objects(out.get(), tup, std::index_sequence_for<Ts...>{})) {
            return nullptr;
        }

        return std::move(out).escape();
    }

    static PyObject* f(std::tuple<Ts...>& tup) {
        auto out = py::scoped_ref(PyTuple_New(sizeof...(Ts)));

        if (!out) {
            return nullptr;
        }

        if (fill_tuple_as_objects(out.get(), tup, std::index_sequence_for<Ts...>{})) {
            return nullptr;
        }

        return std::move(out).escape();
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
