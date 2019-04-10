#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <ostream>
#include <vector>

#include <Python.h>

// numpy expects code is being used in a way that has a 1:1 correspondence from source
// file to Python extension (as a shared object). For unclear reasons, numpy uses it's own
// linking system where the `numpy/arrayobject.h` will put a `static void** PyArray_API =
// nullptr` name into *each* TU that includes it. Many if not all of the API functions in
// numpy are actually macros that resolve to something like: `((PyArrayAPIObject*)
// PyArray_API)->function`. The `import_array()` macro will import (through Python) the
// needed numpy extension modules to get the `PyArray_API` out of a capsule-like object.
//
// This whole system works fine for when a single TU turns into a single object; however,
// the test suite for libpy links all the test files together along with `main.cc` into a
// single program. This has made it very hard to figure out when and how to initialize the
// `PyArray_API` value. Instead, we now set a macro when compiling for the tests
// (`LIBPY_COMPILING_FOR_TESTS`) which will control the `NO_IMPORT_ARRAY` flag. This flag
// tells numpy to declare the `PyArray_API` flag as an `extern "C" void** PyArray_API`,
// meaning we expect to have this symbol defined by another object we are to be linked
// with. In `main.cc` we also set `LIBPY_MAIN` to disable this feature, but instead define
// `PY_ARRAY_UNIQUE_SYMBOL` which causes changes the declaration of `PyArray_API` to
// change to: `#define PyArray_API PY_ARRAY_UNIQUE_SYMBOL` and then `void**
// PyArray_API`. Importantly, this removes the `static` causing the symbol to have
// external linkage. Then, because the tests are declaring the same symbol as extern, they
// will all resolve to the same `PyArray_API` instance and we only need to call
// `import_array` once in `main.cc`.
#if defined(LIBPY_COMPILING_FOR_TESTS) && !defined(LIBPY_MAIN)
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_libpy
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "libpy/any.h"
#include "libpy/array_view.h"
#include "libpy/automethod.h"
#include "libpy/char_sequence.h"
#include "libpy/datetime64.h"
#include "libpy/exception.h"
#include "libpy/scoped_ref.h"
#include "libpy/to_object.h"

namespace py {
inline void ensure_import_array() {
    // If `NO_IMPORT_ARRAY` is defined, this is a nop.
#ifndef NO_IMPORT_ARRAY
    if (PyArray_API) {
        // already imported
        return;
    }

#if PY_MAJOR_VERSION == 2
    []() -> void {
        import_array();
    }();
#else
    // this macro returns NULL in Python 3 so we need to put it in a
    // function to call it to ignore the return statement
    []() -> std::nullptr_t {
        import_array();
        return nullptr;
    }();
#endif
    if (PyErr_Occurred()) {
        throw py::exception{};
    }
#endif
}

/** A strong typedef of npy_bool to not be ambiguous with `unsigned char` but may
    still be used in a vector without the dreaded `std::vector<bool>`.
*/
struct py_bool {
    bool value = false;

    inline py_bool() {}
    inline py_bool(bool v) : value(v) {}
    inline py_bool(npy_bool v) : value(v) {}
    inline py_bool(const py_bool& v) : value(v.value) {}

    inline explicit operator bool() const {
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

template<>
struct new_dtype<std::int8_t> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_INT8);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<>
struct new_dtype<std::int16_t> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_INT16);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<>
struct new_dtype<std::int32_t> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_INT32);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<>
struct new_dtype<std::int64_t> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_INT64);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<>
struct new_dtype<std::uint8_t> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_UINT8);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<>
struct new_dtype<std::uint16_t> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_UINT16);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<>
struct new_dtype<std::uint32_t> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_UINT32);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<>
struct new_dtype<std::uint64_t> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_UINT64);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<>
struct new_dtype<float> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_FLOAT32);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<>
struct new_dtype<double> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_FLOAT64);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<>
struct new_dtype<PyObject*> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_OBJECT);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

template<typename T>
struct new_dtype<scoped_ref<T>> : new_dtype<PyObject*> {};

template<>
struct new_dtype<bool> {
    static PyArray_Descr* get() {
        auto ob = PyArray_DescrFromType(NPY_BOOL);
        if (!ob) {
            return nullptr;
        }
        Py_INCREF(ob);
        return ob;
    }
};

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
scoped_ref<PyArray_Descr> new_dtype() {
    return scoped_ref<PyArray_Descr>(dispatch::new_dtype<T>::get());
}

namespace dispatch {
template<>
struct raise_format<PyArray_Descr*> {
    static std::ostream& f(std::ostream& out, PyArray_Descr* value) {
        return raise_format<PyObject*>::f(out, reinterpret_cast<PyObject*>(value));
    }
};

/** Lookup the proper any_vtable for the given numpy dtype.

    @param dtype The runtime numpy dtype.
    @return The any_vtable that corresponds to the given dtype.
 */
inline any_vtable dtype_to_vtable(PyArray_Descr* dtype) {
    switch (dtype->type_num) {
    case NPY_BOOL:
        return any_vtable::make<py_bool>();
    case NPY_INT8:
        return any_vtable::make<std::int8_t>();
    case NPY_INT16:
        return any_vtable::make<std::int16_t>();
    case NPY_INT32:
        return any_vtable::make<std::int32_t>();
    case NPY_INT64:
        return any_vtable::make<std::int64_t>();
    case NPY_UINT8:
        return any_vtable::make<std::uint8_t>();
    case NPY_UINT16:
        return any_vtable::make<std::uint16_t>();
    case NPY_UINT32:
        return any_vtable::make<std::uint32_t>();
    case NPY_UINT64:
        return any_vtable::make<std::uint64_t>();
    case NPY_FLOAT32:
        return any_vtable::make<float>();
    case NPY_FLOAT64:
        return any_vtable::make<double>();
    case NPY_DATETIME:
        switch (auto unit = reinterpret_cast<PyArray_DatetimeDTypeMetaData*>(
                                dtype->c_metadata)
                                ->meta.base) {
        case py_chrono_unit_to_numpy_unit<py::chrono::ns>:
            return any_vtable::make<py::datetime64<py::chrono::ns>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::us>:
            return any_vtable::make<py::datetime64<py::chrono::us>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::ms>:
            return any_vtable::make<py::datetime64<py::chrono::ms>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::s>:
            return any_vtable::make<py::datetime64<py::chrono::s>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::m>:
            return any_vtable::make<py::datetime64<py::chrono::m>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::h>:
            return any_vtable::make<py::datetime64<py::chrono::h>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::D>:
            return any_vtable::make<py::datetime64<py::chrono::D>>();
        case NPY_FR_GENERIC:
            throw exception(PyExc_TypeError, "cannot adapt unitless datetime");
        default:
            throw exception(PyExc_TypeError, "unknown datetime unit: ", unit);
        }
    case NPY_OBJECT:
        return any_vtable::make<scoped_ref<>>();
    }

    throw exception(PyExc_TypeError,
                    "cannot create an any ref view over an ndarray of dtype: ",
                    reinterpret_cast<PyObject*>(dtype));
}

/** Lookup the proper dtype for the given vtable.

    @param vtable The runtime vtable.
    @return The numpy dtype that corresponds to the given vtable.
 */
inline scoped_ref<PyArray_Descr> vtable_to_dtype(const any_vtable& vtable) {
    if (vtable == any_vtable::make<py_bool>())
        return py::new_dtype<py_bool>();
    if (vtable == any_vtable::make<std::int8_t>())
        return py::new_dtype<std::int8_t>();
    if (vtable == any_vtable::make<std::int16_t>())
        return py::new_dtype<std::int16_t>();
    if (vtable == any_vtable::make<std::int32_t>())
        return py::new_dtype<std::int32_t>();
    if (vtable == any_vtable::make<std::int64_t>())
        return py::new_dtype<std::int64_t>();
    if (vtable == any_vtable::make<std::uint8_t>())
        return py::new_dtype<uint8_t>();
    if (vtable == any_vtable::make<std::uint16_t>())
        return py::new_dtype<uint16_t>();
    if (vtable == any_vtable::make<std::uint32_t>())
        return py::new_dtype<uint32_t>();
    if (vtable == any_vtable::make<std::uint64_t>())
        return py::new_dtype<uint64_t>();
    if (vtable == any_vtable::make<float>())
        return py::new_dtype<float>();
    if (vtable == any_vtable::make<double>())
        return py::new_dtype<double>();
    if (vtable == any_vtable::make<py::datetime64<py::chrono::ns>>())
        return py::new_dtype<py::datetime64<py::chrono::ns>>();
    if (vtable == any_vtable::make<py::datetime64<py::chrono::us>>())
        return py::new_dtype<py::datetime64<py::chrono::us>>();
    if (vtable == any_vtable::make<py::datetime64<py::chrono::ms>>())
        return py::new_dtype<py::datetime64<py::chrono::ms>>();
    if (vtable == any_vtable::make<py::datetime64<py::chrono::s>>())
        return py::new_dtype<py::datetime64<py::chrono::s>>();
    if (vtable == any_vtable::make<py::datetime64<py::chrono::m>>())
        return py::new_dtype<py::datetime64<py::chrono::m>>();
    if (vtable == any_vtable::make<py::datetime64<py::chrono::h>>())
        return py::new_dtype<py::datetime64<py::chrono::h>>();
    if (vtable == any_vtable::make<py::datetime64<py::chrono::D>>())
        return py::new_dtype<py::datetime64<py::chrono::D>>();
    if (vtable == any_vtable::make<scoped_ref<>>())
        return py::new_dtype<PyObject*>();

    throw exception(PyExc_TypeError,
                    "cannot create an dtype from the vtable for type: ",
                    vtable.type_name().get());
}

template<typename T, std::size_t ndim>
struct from_object<ndarray_view<T, ndim>> {
    static ndarray_view<T, ndim> f(PyObject* ob) {
        if (!PyArray_Check(ob)) {
            throw invalid_conversion::make<ndarray_view<T, ndim>>(ob);
        }

        auto array = reinterpret_cast<PyArrayObject*>(ob);

        if (PyArray_NDIM(array) != ndim) {
            throw exception(PyExc_TypeError,
                            "argument must be a ",
                            ndim,
                            " dimensional array, got ndim=",
                            PyArray_NDIM(array));
        }

        std::array<std::size_t, ndim> shape{0};
        std::array<std::int64_t, ndim> strides{0};

        std::copy_n(PyArray_SHAPE(array), ndim, shape.begin());
        std::copy_n(PyArray_STRIDES(array), ndim, strides.begin());

        auto given_dtype = PyArray_DTYPE(array);

        if constexpr (std::is_same_v<T, py::any_ref> || std::is_same_v<T, py::any_cref>) {
            any_vtable vtable = dtype_to_vtable(given_dtype);
            return ndarray_view<T, ndim>(PyArray_BYTES(array), shape, strides, vtable);
        }
        else {
            // note: This is a "constexpr else", removing and unindenting this
            // else block would have semantic meaning and be incorrect. This
            // branch is only expanded when the above test is false; if the
            // "else" is removed, it will always be expanded.

            auto expected_dtype = py::new_dtype<T>();
            if (!given_dtype) {
                throw exception{};
            }

            if (!PyObject_RichCompareBool(reinterpret_cast<PyObject*>(given_dtype),
                                          reinterpret_cast<PyObject*>(
                                              expected_dtype.get()),
                                          Py_EQ)) {
                throw exception(PyExc_TypeError,
                                "expected array of dtype: ",
                                expected_dtype,
                                ", got array of type: ",
                                given_dtype);
            }

            return ndarray_view<T, ndim>(PyArray_BYTES(array), shape, strides);
        }
    }
};

template<typename D>
struct from_object<datetime64<D>> {
    static datetime64<D> f(PyObject* ob) {
        if (!PyArray_CheckScalar(ob)) {
            throw invalid_conversion::make<datetime64<D>>(ob);
        }

        scoped_ref array(
            reinterpret_cast<PyArrayObject*>(PyArray_FromScalar(ob, nullptr)));

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
        PyArray_ScalarAsCtype(ob, &out);
        return out;
    }
};

template<>
struct from_object<py_bool> {
    static py_bool f(PyObject* ob) {
        if (!PyBool_Check(ob)) {
            throw invalid_conversion::make<py_bool>(ob);
        }

        return ob == Py_True;
    }
};

/** Convert a datetime64 in to a numpy array scalar.
 */
template<typename D>
struct to_object<datetime64<D>> {
    static PyObject* f(datetime64<D> dt) {
        auto descr = py::new_dtype<datetime64<D>>();
        if (!descr) {
            return nullptr;
        }
        std::int64_t as_int = static_cast<std::int64_t>(dt);
        return PyArray_Scalar(&as_int, descr.get(), nullptr);
    }
};

/** Convert an any_ref into a numpy array scalar.
 */
template<>
struct to_object<py::any_ref> {
    static PyObject* f(const py::any_ref& ref) {
        auto descr = py::dispatch::vtable_to_dtype(ref.vtable());
        if (!descr) {
            return nullptr;
        }

        // `PyArray_Scalar` copies the input, but incorrectly takes a non-const void*
        return PyArray_Scalar(const_cast<void*>(ref.addr()), descr.get(), nullptr);
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
    static std::optional<std::tuple<scoped_ref<>, C&>> alloc(C&& container) {
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

        return std::make_tuple(scoped_ref(pycapsule), std::ref(cap->container));
    }
};
}  // namespace detail

/** Convert a container into a numpy `ndarray`. This steals the underlying
    buffer from the values array.

    @param values The container to convert. On success this buffer gets moved from
           and will be invalidated.
    @return An `ndarray` from the values.
 */
template<typename C, std::size_t ndim>
scoped_ref<> move_to_numpy_array(C&& values,
                                 py::scoped_ref<PyArray_Descr> descr,
                                 const std::array<std::size_t, ndim>& shape,
                                 const std::array<std::int64_t, ndim>& strides) {
    auto maybe_capsule = detail::capsule<C>::alloc(std::move(values));

    if (!maybe_capsule) {
        // we failed to allocate the python capsule, reraise
        return nullptr;
    }

    auto& [pycapsule, container] = *maybe_capsule;

    scoped_ref arr(PyArray_NewFromDescr(
        &PyArray_Type,
        descr.get(),
        ndim,
        const_cast<npy_intp*>(reinterpret_cast<const npy_intp*>(shape.data())),
        const_cast<npy_intp*>(reinterpret_cast<const npy_intp*>(strides.data())),
        container.data(),
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
scoped_ref<> move_to_numpy_array(std::vector<T>&& values) {
    auto descr = new_dtype<T>();
    if (!descr) {
        return nullptr;
    }
    return move_to_numpy_array<std::vector<T>, 1>(std::move(values),
                                                  descr,
                                                  {values.size()},
                                                  {sizeof(T)});
}

inline scoped_ref<> move_to_numpy_array(py::any_vector&& values) {

    auto descr = py::dispatch::vtable_to_dtype(values.vtable());
    if (!descr) {
        return nullptr;
    }
    return move_to_numpy_array<py::any_vector, 1>(std::move(values),
                                                  descr,
                                                  {values.size()},
                                                  {static_cast<std::int64_t>(
                                                      values.vtable().size())});
}
}  // namespace py
