#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <ostream>
#include <type_traits>
#include <vector>

#include "libpy/borrowed_ref.h"
#include "libpy/buffer.h"
#include "libpy/char_sequence.h"
#include "libpy/datetime64.h"
#include "libpy/detail/numpy.h"
#include "libpy/exception.h"
#include "libpy/object_map_key.h"
#include "libpy/owned_ref.h"
#include "libpy/to_object.h"

/* Macro for ensuring that the Numpy Array API is initialized in a non-extension
   translation unit. It works by declaring an anonymous namespace containing a
   default-constructed instance of an object whose constructor initializes the numpy
   api. This ensures that the numpy api is initialized within the translation unit

   This macro should be invoked at the top-level scope of any **non-extension** .cc file
   that uses the numpy array API (i.e. any of the functions listed in
   https://docs.scipy.org/doc/numpy/reference/c-api.array.html).

   Extension files (i.e., files that correspond to importable Python modules) should call
   import_array() as documented in
   https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.import_array.

   If you're seeing mysterious segfault calling numpy functions from a libpy .cc file, you
   probably need to add this to a .cc file.
*/
#ifdef NO_IMPORT_ARRAY
#define IMPORT_ARRAY_MODULE_SCOPE()
#else
#define IMPORT_ARRAY_MODULE_SCOPE()                                                      \
    namespace {                                                                          \
    struct importer {                                                                    \
        importer() {                                                                     \
            if (!PyArray_API) {                                                          \
                import_array();                                                          \
                if (PyErr_Occurred()) {                                                  \
                    throw py::exception{};                                               \
                }                                                                        \
            }                                                                            \
        }                                                                                \
    };                                                                                   \
    importer ensure_array_imported;                                                      \
    }
#endif

namespace py {

/** A strong typedef of npy_bool to not be ambiguous with `unsigned char` but may
    still be used in a vector without the dreaded `std::vector<bool>`.
*/
struct py_bool {
    bool value = false;

    inline py_bool() {}
    inline py_bool(bool v) : value(v) {}
    inline py_bool(npy_bool v) : value(v) {}
    inline py_bool(const py_bool& v) : value(v.value) {}

    inline py_bool& operator=(const py_bool&) = default;

    inline operator bool() const {
        return value;
    }

    inline bool operator==(const py_bool& other) const {
        return value == other.value;
    }

    inline bool operator!=(const py_bool& other) const {
        return value != other.value;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const py_bool& value) {
    return stream << value.value;
}

template<>
inline constexpr char buffer_format<py::py_bool> = '?';

namespace detail {
template<typename D>
struct py_chrono_unit_to_numpy_unit;

template<>
struct py_chrono_unit_to_numpy_unit<py::chrono::ns> {
    static constexpr auto value = NPY_FR_ns;
};

template<>
struct py_chrono_unit_to_numpy_unit<py::chrono::us> {
    static constexpr auto value = NPY_FR_us;
};

template<>
struct py_chrono_unit_to_numpy_unit<py::chrono::ms> {
    static constexpr auto value = NPY_FR_ms;
};

template<>
struct py_chrono_unit_to_numpy_unit<py::chrono::s> {
    static constexpr auto value = NPY_FR_s;
};

template<>
struct py_chrono_unit_to_numpy_unit<py::chrono::m> {
    static constexpr auto value = NPY_FR_m;
};

template<>
struct py_chrono_unit_to_numpy_unit<py::chrono::h> {
    static constexpr auto value = NPY_FR_h;
};

template<>
struct py_chrono_unit_to_numpy_unit<py::chrono::D> {
    static constexpr auto value = NPY_FR_D;
};
}  // namespace detail

template<typename D>
constexpr auto py_chrono_unit_to_numpy_unit =
    detail::py_chrono_unit_to_numpy_unit<D>::value;

namespace dispatch {
template<typename T>
struct new_dtype;

template<auto typecode>
struct new_dtype_from_typecode {
    static PyArray_Descr* get() {
        PyArray_Descr* out = PyArray_DescrFromType(typecode);
        Py_XINCREF(out);
        return out;
    }
};

namespace detail {
template<bool is_signed, std::size_t size>
struct integral_typecode;

template<>
struct integral_typecode<true, 1> {
    static constexpr int value = NPY_INT8;
};

template<>
struct integral_typecode<true, 2> {
    static constexpr int value = NPY_INT16;
};

template<>
struct integral_typecode<true, 4> {
    static constexpr int value = NPY_INT32;
};

template<>
struct integral_typecode<true, 8> {
    static constexpr int value = NPY_INT64;
};

template<>
struct integral_typecode<false, 1> {
    static constexpr int value = NPY_UINT8;
};

template<>
struct integral_typecode<false, 2> {
    static constexpr int value = NPY_UINT16;
};

template<>
struct integral_typecode<false, 4> {
    static constexpr int value = NPY_UINT32;
};

template<>
struct integral_typecode<false, 8> {
    static constexpr int value = NPY_UINT64;
};

template<typename T>
using new_dtype_integral =
    new_dtype_from_typecode<integral_typecode<std::is_signed_v<T>, sizeof(T)>::value>;
}  // namespace detail

template<>
struct new_dtype<char> : detail::new_dtype_integral<char> {};

template<>
struct new_dtype<signed char> : detail::new_dtype_integral<signed char> {};

template<>
struct new_dtype<unsigned char> : detail::new_dtype_integral<unsigned char> {};

template<>
struct new_dtype<signed short> : detail::new_dtype_integral<signed short> {};

template<>
struct new_dtype<unsigned short> : detail::new_dtype_integral<unsigned short> {};

template<>
struct new_dtype<signed int> : detail::new_dtype_integral<signed int> {};

template<>
struct new_dtype<unsigned int> : detail::new_dtype_integral<unsigned int> {};

template<>
struct new_dtype<signed long> : detail::new_dtype_integral<signed long> {};

template<>
struct new_dtype<unsigned long> : detail::new_dtype_integral<unsigned long> {};

template<>
struct new_dtype<signed long long> : detail::new_dtype_integral<signed long long> {};

template<>
struct new_dtype<unsigned long long> : detail::new_dtype_integral<unsigned long long> {};

template<>
struct new_dtype<float> : new_dtype_from_typecode<NPY_FLOAT32> {};

template<>
struct new_dtype<double> : new_dtype_from_typecode<NPY_FLOAT64> {};

template<>
struct new_dtype<PyObject*> : new_dtype_from_typecode<NPY_OBJECT> {};

template<typename T>
struct new_dtype<py::borrowed_ref<T>> : new_dtype_from_typecode<NPY_OBJECT> {};

template<typename T>
struct new_dtype<owned_ref<T>> : new_dtype<PyObject*> {};

template<>
struct new_dtype<object_map_key> : new_dtype<PyObject*> {};

template<>
struct new_dtype<bool> : new_dtype_from_typecode<NPY_BOOL> {};

template<>
struct new_dtype<py_bool> : public new_dtype<bool> {};

template<typename D>
struct new_dtype<datetime64<D>> {
    static PyArray_Descr* get() {
        PyArray_Descr* out = PyArray_DescrNewFromType(NPY_DATETIME);
        if (!out) {
            return nullptr;
        }

        auto dt_meta = reinterpret_cast<PyArray_DatetimeDTypeMetaData*>(out->c_metadata);
        dt_meta->meta.base = py_chrono_unit_to_numpy_unit<D>;
        dt_meta->meta.num = 1;
        return out;
    }
};

template<std::size_t size>
struct new_dtype<std::array<char, size>> {
    static PyArray_Descr* get() {
        PyArray_Descr* descr = PyArray_DescrNewFromType(NPY_STRING);
        if (descr) {
            descr->elsize = size;
        }
        return descr;
    }
};
}  // namespace dispatch

/** Create a new numpy dtype object for a given C++ type.

    @tparam T The C++ type to get the numpy dtype of.
    @return A new reference to a numpy dtype for the given type.
 */
template<typename T>
owned_ref<PyArray_Descr> new_dtype() {
    return owned_ref<PyArray_Descr>(dispatch::new_dtype<T>::get());
}

namespace detail {
template<typename T>
struct has_new_dtype {
private:
    template<typename U>
    static decltype(py::dispatch::new_dtype<U>::get(), std::true_type{}) test(int);

    template<typename>
    static std::false_type test(long);

public:
    static constexpr bool value = std::is_same_v<decltype(test<T>(0)), std::true_type>;
};
}  // namespace detail

/** Compile time boolean to detect if `new_dtype` works for a given type. This exists to
    make it easier to use `if constexpr` to test this condition instead of using more
    complicated SFINAE.
 */
template<typename T>
constexpr bool has_new_dtype = detail::has_new_dtype<T>::value;

namespace dispatch {
template<>
struct raise_format<PyArray_Descr*> {
    static std::ostream& f(std::ostream& out, PyArray_Descr* value) {
        return raise_format<PyObject*>::f(out, reinterpret_cast<PyObject*>(value));
    }
};

template<>
struct from_object<py_bool> {
    static py_bool f(py::borrowed_ref<> ob) {
        if (!PyBool_Check(ob.get())) {
            throw invalid_conversion::make<py_bool>(ob);
        }

        return ob == Py_True;
    }
};

template<>
struct to_object<py_bool> {
    static py::owned_ref<> f(py_bool v) {
        return py::owned_ref{PyBool_FromLong(v.value)};
    }
};

/** Convert a datetime64 in to a numpy array scalar.
 */
template<typename unit>
struct to_object<datetime64<unit>> {
    static py::owned_ref<> f(const datetime64<unit>& dt) {
        auto descr = py::new_dtype<datetime64<unit>>();
        if (!descr) {
            return nullptr;
        }
        std::int64_t as_int = static_cast<std::int64_t>(dt);
        return py::owned_ref{PyArray_Scalar(&as_int, descr.get(), nullptr)};
    }
};

template<typename D>
struct from_object<datetime64<D>> {
    static datetime64<D> f(py::borrowed_ref<> ob) {
        if (!PyArray_CheckScalar(ob.get())) {
            throw invalid_conversion::make<datetime64<D>>(ob);
        }

        py::owned_ref array(
            reinterpret_cast<PyArrayObject*>(PyArray_FromScalar(ob.get(), nullptr)));

        auto dtype = py::new_dtype<datetime64<D>>();
        if (!dtype) {
            throw exception{};
        }

        if (!PyObject_RichCompareBool(reinterpret_cast<PyObject*>(
                                          PyArray_DTYPE(array.get())),
                                      reinterpret_cast<PyObject*>(dtype.get()),
                                      Py_EQ)) {
            throw exception(PyExc_TypeError,
                            "expected array of dtype: ",
                            dtype,
                            ", got array of type: ",
                            PyArray_DTYPE(array.get()));
        }

        datetime64<D> out;
        PyArray_ScalarAsCtype(ob.get(), &out);
        return out;
    }
};
}  // namespace dispatch

namespace detail {
/** A capsule to add Python reference counting to a contiguous container. This is
    used to allow Python to manage the lifetimes of vectors fed to
    `move_to_numpy_array`.

    For more info on ``PyCapsule`` objects, see:
    https://docs.python.org/3/c-api/capsule.html
 */
template<typename C>
struct capsule {
private:
    /** The backing vector, this should not be mutated during the capsule's
        lifetime by anyone but the owning `ndarray`.
     */
    C container;

    /** Free this object with PyMem_Free after properly destructing members
     */
    void free() {
        // inplace-destruct our capsule object
        this->~capsule();
        PyMem_Free(this);
    }

    /** The function to pass as the deallocate function for `PyCapsule_New`.
        @param pycapsule A python capsule object.
    */
    static void py_capsule_dealloc(PyObject* pycapsule) {
        capsule* cap = reinterpret_cast<capsule*>(
            PyCapsule_GetPointer(pycapsule, nullptr));
        if (cap) {
            cap->free();
        }
    }

    /** Private constructor because you should only ever create this with
        `alloc` which will box this in a `PyCapsuleObject*`.
    */
    capsule(C&& container) : container(std::move(container)) {}

public:
    /** Allocate a new capsule object on the python heap.

        @param vector The vector to store in the capsule.
        @return Either `std::nullopt` with a python exception raised, or a
                `std::tuple` of the python capsule object and the moved vector
                it is refcounting for.
    */
    static std::optional<std::tuple<owned_ref<>, C&>> alloc(C&& container) {
        capsule* cap;
        if (!(cap = reinterpret_cast<capsule*>(PyMem_Malloc(sizeof(capsule))))) {
            return {};
        }
        // placement move construct our container in the memory alloced with
        // PyMem_Malloc
        new (cap) capsule(std::move(container));

        PyObject* pycapsule = PyCapsule_New(cap, nullptr, capsule::py_capsule_dealloc);
        if (!pycapsule) {
            cap->free();
            return {};
        }

        return std::make_tuple(owned_ref(pycapsule), std::ref(cap->container));
    }
};
}  // namespace detail

class any_vector;

/** Convert a container into a numpy `ndarray`. This steals the underlying
    buffer from the values array.

    @param values The container to convert. On success this buffer gets moved from
           and will be invalidated.
    @return An `ndarray` from the values.
 */
template<typename C, std::size_t ndim>
owned_ref<> move_to_numpy_array(C&& values,
                                py::owned_ref<PyArray_Descr> descr,
                                const std::array<std::size_t, ndim>& shape,
                                const std::array<std::int64_t, ndim>& strides) {
    auto maybe_capsule = detail::capsule<C>::alloc(std::move(values));

    if (!maybe_capsule) {
        // we failed to allocate the python capsule, reraise
        return nullptr;
    }

    auto& [pycapsule, container] = *maybe_capsule;

    std::byte* data;
    if constexpr (std::is_same_v<C, py::any_vector>) {
        data = container.buffer();
    }
    else {
        data = reinterpret_cast<std::byte*>(container.data());
    }

    owned_ref arr(PyArray_NewFromDescr(
        &PyArray_Type,
        descr.get(),
        ndim,
        const_cast<npy_intp*>(reinterpret_cast<const npy_intp*>(shape.data())),
        const_cast<npy_intp*>(reinterpret_cast<const npy_intp*>(strides.data())),
        data,
        NPY_ARRAY_CARRAY,
        nullptr));
    if (!arr) {
        return nullptr;
    }

    // `PyArray_NewFromDescr` steals a reference to `descr` on success
    std::move(descr).escape();

    if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(arr.get()),
                              pycapsule.get())) {
        return nullptr;
    }

    // `PyArray_SetBaseObject` steals a reference to `pycapsule` on success
    std::move(pycapsule).escape();

    return arr;
}

template<typename T>
owned_ref<> move_to_numpy_array(std::vector<T>&& values) {
    auto descr = new_dtype<T>();
    if (!descr) {
        return nullptr;
    }
    return move_to_numpy_array<std::vector<T>, 1>(std::move(values),
                                                  std::move(descr),
                                                  {values.size()},
                                                  {sizeof(T)});
}
}  // namespace py

namespace std {
template<>
struct hash<py::py_bool> {
    auto operator()(py::py_bool b) const noexcept {
        return std::hash<bool>{}(static_cast<bool>(b));
    }
};
}  // namespace std
