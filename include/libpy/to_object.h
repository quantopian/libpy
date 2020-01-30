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

#include "libpy/char_sequence.h"
#include "libpy/detail/python.h"
#include "libpy/scoped_ref.h"
#include "libpy/singletons.h"
#include "libpy/str_convert.h"

namespace py {
namespace dispatch {
/** Dispatch struct for defining overrides for `py::to_object`.

    To add a new dispatch, add an explicit template specialization for the type
    to convert with a static member function `f` which accepts the dispatched
    type and returns a `py::scoped_ref<>`.
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
py::scoped_ref<> to_object(T&& ob) {
    using underlying_type = std::remove_cv_t<std::remove_reference_t<T>>;
    return dispatch::to_object<underlying_type>::f(std::forward<T>(ob));
}

namespace dispatch {
template<typename C>
py::scoped_ref<> sequence_to_list(const C& v) {
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

    return out;
}

template<std::size_t n>
struct to_object<std::array<char, n>> {
    static py::scoped_ref<> f(const std::array<char, n>& cs) {
        return py::scoped_ref{PyBytes_FromStringAndSize(cs.data(), n)};
    }
};

template<>
struct to_object<char> {
    static py::scoped_ref<> f(char c) {
        return py::scoped_ref{PyBytes_FromStringAndSize(&c, 1)};
    }
};

template<typename T, std::size_t n>
struct to_object<std::array<T, n>> {
    static py::scoped_ref<> f(const std::array<T, n>& v) {
        return sequence_to_list(v);
    }
};

/** Identity conversion for `scoped_ref`.
 */
template<typename T>
struct to_object<scoped_ref<T>> {
    static py::scoped_ref<> f(const scoped_ref<T>& ob) {
        return ob;
    }
};

template<>
struct to_object<std::string> {
    static py::scoped_ref<> f(const std::string& cs) {
        return py::scoped_ref{PyBytes_FromStringAndSize(cs.data(), cs.size())};
    }
};

template<>
struct to_object<std::string_view> {
    static py::scoped_ref<> f(const std::string_view& cs) {
        return py::scoped_ref{PyBytes_FromStringAndSize(cs.data(), cs.size())};
    }
};

template<std::size_t n>
struct to_object<char[n]> {
    static py::scoped_ref<> f(const char* cs) {
        return py::scoped_ref{PyBytes_FromStringAndSize(cs, n - 1)};
    }
};

template<>
struct to_object<bool> {
    static py::scoped_ref<> f(bool value) {
        return py::scoped_ref{PyBool_FromLong(value)};
    }
};

namespace detail {
template<typename T>
struct int_to_object {
    static py::scoped_ref<> f(T value) {

        // convert the object to the widest type for the given signedness
        if constexpr (std::is_signed_v<T>) {
            return py::scoped_ref{PyLong_FromLongLong(value)};
        }
        else {
            return py::scoped_ref{PyLong_FromUnsignedLongLong(value)};
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
    static py::scoped_ref<> f(float value) {
        return py::scoped_ref{PyFloat_FromDouble(value)};
    }
};

template<>
struct to_object<double> {
    static py::scoped_ref<> f(double value) {
        return py::scoped_ref{PyFloat_FromDouble(value)};
    }
};

template<typename M>
struct map_to_object {
    static py::scoped_ref<> f(const M& m) {
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

        return out;
    }
};

template<typename K, typename V, typename Hash, typename KeyEqual>
struct to_object<std::unordered_map<K, V, Hash, KeyEqual>>
    : public map_to_object<std::unordered_map<K, V, Hash, KeyEqual>> {};

template<typename T>
struct to_object<std::vector<T>> {
    static py::scoped_ref<> f(const std::vector<T>& v) {
        return sequence_to_list(v);
    }
};

template<typename S>
struct set_to_object {
    static py::scoped_ref<> f(const S& s) {
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

        return out;
    }
};

template<typename T>
struct to_object<std::unordered_set<T>> : set_to_object<std::unordered_set<T>> {};

template<typename... Ts>
struct to_object<std::tuple<Ts...>> {
private:
    template<typename T, std::size_t... Ix>
    static bool
    fill_tuple_as_objects(py::borrowed_ref<> out, T&& tup, std::index_sequence<Ix...>) {
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
    static py::scoped_ref<> f(const std::tuple<Ts...>& tup) {
        py::scoped_ref out(PyTuple_New(sizeof...(Ts)));

        if (!out) {
            return nullptr;
        }

        if (fill_tuple_as_objects(out.get(), tup, std::index_sequence_for<Ts...>{})) {
            return nullptr;
        }

        return out;
    }
};

template<typename T>
struct to_object<std::optional<T>> {
    static py::scoped_ref<> f(const std::optional<T>& maybe_value) {
        if (!maybe_value) {
            return py::none;
        }

        return py::to_object(*maybe_value);
    }
};
}  // namespace dispatch
}  // namespace py
