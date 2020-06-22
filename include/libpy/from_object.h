#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "libpy/borrowed_ref.h"
#include "libpy/demangle.h"
#include "libpy/detail/autoclass_cache.h"
#include "libpy/detail/autoclass_object.h"
#include "libpy/detail/python.h"
#include "libpy/dict_range.h"
#include "libpy/exception.h"
#include "libpy/owned_ref.h"
#include "libpy/util.h"

namespace py {
/** Exception raised when an invalid `py::from_object` call is performed.
 */
class invalid_conversion : public py::exception {
public:
    inline invalid_conversion(const std::string& msg)
        : py::exception(PyExc_TypeError, msg) {}

    template<typename ConvertTo>
    static invalid_conversion make(py::borrowed_ref<> ob) {
        py::owned_ref repr(PyObject_Repr(ob.get()));
        if (!repr) {
            throw py::exception("failed to call repr on ob");
        }
        const char* data = py::util::pystring_to_cstring(repr);
        if (!data) {
            throw py::exception("failed to get utf8 string from repr result");
        }

        std::stringstream s;
        s << "failed to convert Python object of type " << Py_TYPE(ob.get())->tp_name
          << " to a C++ object of type " << py::util::type_name<ConvertTo>()
          << ": ob=" << data;
        return invalid_conversion(s.str());
    }
};

namespace dispatch {
/** Dispatch struct for defining overrides for `py::from_object`.

    To add a new dispatch, add an explicit template specialization for the type
    to convert with a static member function `f` which accepts
    `py::borrowed_ref<>` and returns a new object of type `T`.
 */
template<typename T>
struct from_object;
}  // namespace dispatch

/** Convert a Python object into a C++ object recursively.

    @param ob The object to convert
    @return `ob` as a C++ object.
    @see py::dispatch::from_object
 */
template<typename T>
decltype(auto) from_object(py::borrowed_ref<> ob) {
    return dispatch::from_object<T>::f(ob);
}

namespace detail {
template<typename T>
struct has_from_object {
private:
    template<typename U>
    static decltype(py::dispatch::from_object<U>::f(std::declval<py::borrowed_ref<>>()),
                    std::true_type{})
    test(int);

    template<typename>
    static std::false_type test(long);

public:
    static constexpr bool value = std::is_same_v<decltype(test<T>(0)), std::true_type>;
};
}  // namespace detail

/** Compile time boolean to detect if `from_object` works for a given type. This exists to
    make it easier to use `if constexpr` to test this condition instead of using more
    complicated SFINAE.
 */
template<typename T>
constexpr bool has_from_object = detail::has_from_object<T>::value;

namespace dispatch {

template<typename T>
struct from_object<T&> {
private:
    using mut_T = std::remove_const_t<T>;
    static constexpr bool specialized = std::is_const_v<T> &&
                                        py::has_from_object<mut_T> &&
                                        !py::has_from_object<T>;

public:
    static decltype(auto) f(py::borrowed_ref<> ob) {
        if constexpr (specialized) {
            return py::from_object<mut_T>(ob);
        }
        else {
            auto search = py::detail::autoclass_type_cache.get().find(typeid(mut_T));
            if (search == py::detail::autoclass_type_cache.get().end()) {
                throw invalid_conversion::make<T&>(ob);
            }
            int res = PyObject_IsInstance(ob.get(),
                                          static_cast<PyObject*>(
                                              search->second->type));
            if (res < 0) {
                throw py::exception{};
            }
            if (!res) {
                throw invalid_conversion::make<T&>(ob);
            }

            // NOTE: the parentheses change the behavior of `decltype(auto)` to make
            // this resolve to a return type of `T&` instead of `T`
            return (*static_cast<T*>(search->second->unbox(ob)));
        }
    }
};

template<typename T>
struct from_object<std::reference_wrapper<T>> {
public:
    static std::reference_wrapper<T> f(borrowed_ref<> ob) {
        return from_object<T&>::f(ob);
    }
};

template<std::size_t n>
struct from_object<std::array<char, n>> {
    static std::array<char, n> f(py::borrowed_ref<> cs) {
        std::array<char, n> out = {0};

        const char* data;
        Py_ssize_t size;

        if (PyBytes_Check(cs.get())) {
            size = PyBytes_GET_SIZE(cs.get());
            data = PyBytes_AS_STRING(cs.get());
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
    static char f(py::borrowed_ref<> cs) {
        const char* data;
        Py_ssize_t size;

        if (PyBytes_Check(cs.get())) {
            size = PyBytes_GET_SIZE(cs.get());
            data = PyBytes_AS_STRING(cs.get());
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
    static PyObject* f(py::borrowed_ref<> ob) {
        return ob.get();
    }
};

/** Identity conversion for `borrowed_ref`.
 */
template<typename T>
struct from_object<py::borrowed_ref<T>> {
    static py::borrowed_ref<T> f(py::borrowed_ref<> ob) {
        return reinterpret_cast<T*>(ob.get());
    }
};

/** Identity conversion for `owned_ref`.
 */
template<typename T>
struct from_object<py::owned_ref<T>> {
    static owned_ref<T> f(py::borrowed_ref<> ob) {
        return owned_ref<T>::new_reference(ob);
    }
};

template<>
struct from_object<std::string_view> {
    static std::string_view f(py::borrowed_ref<> cs) {
        const char* data;
        Py_ssize_t size;

        if (PyBytes_Check(cs)) {
            size = PyBytes_GET_SIZE(cs.get());
            data = PyBytes_AS_STRING(cs.get());
        }
        else {
            throw invalid_conversion::make<std::string_view>(cs);
        }

        return std::string_view(data, size);
    }
};

template<>
struct from_object<std::string> {
    static std::string f(py::borrowed_ref<> cs) {
        return std::string(py::from_object<std::string_view>(cs));
    }
};

namespace detail {
template<typename T, bool cond>
constexpr bool defer_check = cond;
}  // namespace detail

/** Get an std::filesystem::path from an object implementing __fspath__

    @param ob Object implementing __fspath__.
    @return A std::filesystem::path
*/
template<>
struct from_object<std::filesystem::path> {
    // make this a template to defer the static_assert check until the function is
    // used.
    template<typename T = void>
    static std::filesystem::path f([[maybe_unused]] py::borrowed_ref<> ob) {
#if PY_VERSION_HEX < 0x03060000
        static_assert(detail::defer_check<T, (PY_VERSION_HEX >= 0x03060000)>,
                      "cannot use fs_path in Python older than 3.6");
        throw std::runtime_error("cannot use fs_path in Python older than 3.6");
#else
        py::owned_ref path_obj{PyOS_FSPath(ob.get())};

        if (!path_obj) {
            throw py::exception{};
        }
        // depending on the implementation of __fspath__ we can get str or bytes
        std::filesystem::path path;
        if (PyBytes_Check(path_obj)) {
            path = py::from_object<std::string>(path_obj);
        }
        else {
            path = py::util::pystring_to_string_view(path_obj);
        }

        return path;
#endif
    }
};

template<>
struct from_object<bool> {
    static bool f(py::borrowed_ref<> value) {
        if (!PyBool_Check(value)) {
            throw invalid_conversion::make<bool>(value);
        }
        return value == Py_True;
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
    static T f(py::borrowed_ref<> value) {
        wide_type wide;

        // convert the object to the widest type for the given signedness
        if constexpr (std::is_signed_v<T>) {
            wide = PyLong_AsLongLong(value.get());
        }
        else {
            wide = PyLong_AsUnsignedLongLong(value.get());
        }

        // check if `value` would overflow `wide_type`
        if (PyErr_Occurred()) {
            throw invalid_conversion::make<T>(value);
        }

        // check if narrowing the wide value to the given type would overflow
        if (wide > std::numeric_limits<T>::max() ||
            wide < std::numeric_limits<T>::min()) {
            py::raise(PyExc_OverflowError) << "converting " << value << " to type "
                                           << py::util::type_name<T>() << " overflows";
            throw invalid_conversion::make<T>(value);
        }
        return wide;
    }
};
}  // namespace detail

template<>
struct from_object<long long> : public detail::int_from_object<long long> {};

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
    static double f(py::borrowed_ref<> value) {
        double out = PyFloat_AsDouble(value.get());
        if (out == -1.0 && PyErr_Occurred()) {
            throw invalid_conversion::make<double>(value);
        }
        return out;
    }
};

template<>
struct from_object<float> {
    static float f(py::borrowed_ref<> value) {
        float out = PyFloat_AsDouble(value.get());
        if (out == -1.0 && PyErr_Occurred()) {
            throw invalid_conversion::make<float>(value);
        }
        return out;
    }
};

template<typename M>
struct map_from_object {
    static M f(py::borrowed_ref<> m) {
        if (!PyDict_Check(m)) {
            throw invalid_conversion::make<M>(m);
        }

        M out;
        for (auto [key, value] : py::dict_range(m)) {
            out.insert({py::from_object<typename M::key_type>(key),
                        py::from_object<typename M::mapped_type>(value)});
        }
        return out;
    }
};

template<typename K, typename V, typename Hash, typename KeyEqual>
struct from_object<std::unordered_map<K, V, Hash, KeyEqual>>
    : map_from_object<std::unordered_map<K, V, Hash, KeyEqual>> {};

template<typename T>
struct from_object<std::vector<T>> {
    static std::vector<T> f(py::borrowed_ref<> v) {
        if (!PyList_Check(v.get())) {
            throw invalid_conversion::make<std::vector<T>>(v);
        }

        std::vector<T> out;

        Py_ssize_t size = PyList_GET_SIZE(v.get());
        for (Py_ssize_t ix = 0; ix < size; ++ix) {
            out.emplace_back(py::from_object<T>(PyList_GET_ITEM(v.get(), ix)));
        }

        return out;
    }
};

template<typename T, std::size_t n>
struct from_object<std::array<T, n>> {
    static std::array<T, n> f(py::borrowed_ref<> v) {
        if (!PyList_Check(v.get())) {
            throw invalid_conversion::make<std::array<T, n>>(v);
        }

        if (PyList_GET_SIZE(v.get()) != static_cast<Py_ssize_t>(n)) {
            throw py::util::formatted_error<invalid_conversion>(
                "list size does not match fixed-size array size: ",
                PyList_GET_SIZE(v.get()),
                " != ",
                n);
        }

        std::array<T, n> out;
        for (Py_ssize_t ix = 0; ix < static_cast<Py_ssize_t>(out.size()); ++ix) {
            out[ix] = py::from_object<T>(PyList_GET_ITEM(v.get(), ix));
        }

        return out;
    }
};

template<typename T>
struct from_object<std::unordered_set<T>> {
    static std::unordered_set<T> f(py::borrowed_ref<> s) {
        if (!(PySet_Check(s.get()) || PyFrozenSet_Check(s.get()))) {
            throw invalid_conversion::make<std::unordered_set<T>>(s);
        }

        std::unordered_set<T> out;

        py::owned_ref it(PyObject_GetIter(s.get()));
        if (!it) {
            throw py::exception("failed to make iterator");
        }

        while (py::owned_ref<> item{PyIter_Next(it.get())}) {
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
    static std::tuple<Ts...> fill_tuple(py::borrowed_ref<> tup,
                                        std::index_sequence<ixs...>) {
        return {py::from_object<Ts>(PyTuple_GET_ITEM(tup.get(), ixs))...};
    }

public:
    static std::tuple<Ts...> f(py::borrowed_ref<> tup) {
        if (!(PyTuple_Check(tup.get()) && PyTuple_GET_SIZE(tup.get()) == sizeof...(Ts))) {
            throw invalid_conversion::make<std::tuple<Ts...>>(tup);
        }

        return fill_tuple(tup, std::index_sequence_for<Ts...>{});
    }
};

template<typename T>
struct from_object<std::optional<T>> {
    static std::optional<T> f(py::borrowed_ref<> maybe_value) {
        if (maybe_value == nullptr || maybe_value == Py_None) {
            return std::nullopt;
        }

        return py::from_object<T>(maybe_value);
    }
};
}  // namespace dispatch
}  // namespace py
