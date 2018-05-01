#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <ostream>
#include <vector>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "libpy/array_view.h"
#include "libpy/automethod.h"
#include "libpy/char_sequence.h"
#include "libpy/datetime64ns.h"
#include "libpy/exception.h"
#include "libpy/scoped_ref.h"
#include "libpy/to_object.h"

namespace py {
/** A strong typedef of npy_bool to not be ambiguous with `unsigned char` but may still
    be used in a vector without the dreaded `std::vector<bool>`.
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

std::ostream& operator<<(std::ostream& stream, const py_bool& value) {
    return stream << value.value;
}

namespace dispatch {
template<typename T>
struct new_dtype;

template<>
struct new_dtype<npy_int8> {
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
struct new_dtype<npy_int16> {
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
struct new_dtype<npy_int32> {
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
struct new_dtype<npy_int64> {
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
struct new_dtype<npy_uint8> {
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
struct new_dtype<npy_uint16> {
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
struct new_dtype<npy_uint32> {
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
struct new_dtype<npy_uint64> {
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
struct new_dtype<npy_float64> {
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

template<>
struct new_dtype<datetime64ns> {
    static PyArray_Descr* get() {
        PyArray_Descr* out = PyArray_DescrNewFromType(NPY_DATETIME);
        if (!out) {
            return nullptr;
        }

        auto dt_meta = reinterpret_cast<PyArray_DatetimeDTypeMetaData*>(out->c_metadata);
        dt_meta->meta.base = NPY_FR_ns;
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
    using fmt = cs::char_sequence<'R'>;

    static auto prepare(PyArray_Descr* ob) {
        return reinterpret_cast<PyObject*>(ob);
    }
};

template<typename T, std::size_t ndim>
struct from_object<ndarray_view<T, ndim>> {
    static ndarray_view<T, ndim> f(PyObject* ob) {
        if (!PyArray_Check(ob)) {
            throw exception(PyExc_TypeError,
                            "argument must be an ndarray, got: ",
                            Py_TYPE(ob));
        }

        auto array = reinterpret_cast<PyArrayObject*>(ob);

        if (PyArray_NDIM(array) != ndim) {
            throw exception(PyExc_TypeError,
                            "argument must be a ",
                            ndim,
                            " dimensional array, got ndim=",
                            PyArray_NDIM(array));
        }

        auto dtype = py::new_dtype<T>();
        if (!dtype) {
            throw exception{};
        }

        if (!PyObject_RichCompareBool(reinterpret_cast<PyObject*>(PyArray_DTYPE(array)),
                                      reinterpret_cast<PyObject*>(dtype.get()),
                                      Py_EQ)) {
            throw exception(PyExc_TypeError,
                            "expected array of dtype: ",
                            dtype,
                            ", got array of type: ",
                            PyArray_DTYPE(array));
        }

        std::array<std::size_t, ndim> shape{0};
        std::array<std::int64_t, ndim> strides{0};

        std::copy_n(PyArray_SHAPE(array), ndim, shape.begin());
        std::copy_n(PyArray_STRIDES(array), ndim, strides.begin());

        return ndarray_view<T, ndim>(PyArray_BYTES(array), shape, strides);
    }
};

template<>
struct from_object<datetime64ns> {
    static datetime64ns f(PyObject* ob) {
        if (!PyArray_CheckScalar(ob)) {
            throw exception(PyExc_TypeError,
                            "argument must be an array scalar, got:",
                            Py_TYPE(ob));
        }

        auto array = scoped_ref(
            reinterpret_cast<PyArrayObject*>(PyArray_FromScalar(ob, nullptr)));

        auto dtype = py::new_dtype<datetime64ns>();
        if (!dtype) {
            throw exception{};
        }

        if (!PyObject_RichCompareBool(reinterpret_cast<PyObject*>(PyArray_DTYPE(array.get())),
                                      reinterpret_cast<PyObject*>(dtype.get()),
                                      Py_EQ)) {
            throw exception(PyExc_TypeError,
                            "expected array of dtype: ",
                            dtype,
                            ", got array of type: ",
                            PyArray_DTYPE(array.get()));
        }

        datetime64ns out;
        PyArray_ScalarAsCtype(ob, &out);
        return out;
    }
};

/** Convert a datetime64ns in to a numpy array scalar.
 */
template<>
struct to_object<datetime64ns> {
    static PyObject* f(datetime64ns dt) {
        auto descr = py::new_dtype<datetime64ns>();
        if (!descr) {
            return nullptr;
        }
        std::int64_t as_int = static_cast<std::int64_t>(dt);
        return PyArray_Scalar(&as_int, descr.get(), NULL);
    }
};
}  // namespace dispatch

namespace detail {
/** A capsule to add Python reference counting to a `std::vector`. This is
    used to allow Python to manage the lifetimes of vectors fed to
    `move_to_numpy_array`.

    For more info on ``PyCapsule`` objects, see:
    https://docs.python.org/3/c-api/capsule.html
 */
template<typename T>
struct capsule {
private:
    /** The backing vector, this should not be mutated during the capsule's
        lifetime by anyone but the owning `ndarray`.
     */
    std::vector<T> vector;

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
    capsule(std::vector<T>&& vector) : vector(std::move(vector)) {}

public:
    /** Allocate a new capsule object on the python heap.

        @param vector The vector to store in the capsule.
        @return Either `std::nullopt` with a python exception raised, or a
                `std::tuple` of the python capsule object and the moved vector
                it is refcounting for.
    */
    static std::optional<std::tuple<scoped_ref<PyObject>, std::vector<T>&>>
    alloc(std::vector<T>&& vector) {
        capsule* cap;
        if (!(cap = reinterpret_cast<capsule*>(PyMem_Malloc(sizeof(capsule))))) {
            return {};
        }
        // placement move construct our vector in the memory alloced with
        // PyMem_Malloc
        new (cap) capsule(std::move(vector));

        PyObject* pycapsule = PyCapsule_New(cap, nullptr, capsule::py_capsule_dealloc);
        if (!pycapsule) {
            cap->free();
            return {};
        }

        return std::make_tuple(scoped_ref(pycapsule), std::ref(cap->vector));
    }
};
}  // namespace detail

/** Convert a `py::vector<T>` into a numpy `ndarray`. This steals the underlying
    buffer from the values array.

    @param values The vector to convert. On success this buffer gets moved from
           and will be invalidated.
    @return An `ndarray` from the values.
 */
template<typename T, std::size_t ndim>
scoped_ref<PyObject> move_to_numpy_array(std::vector<T>&& values,
                                         const std::array<std::size_t, ndim>& shape,
                                         const std::array<std::int64_t, ndim>& strides) {
    auto maybe_capsule = detail::capsule<T>::alloc(std::move(values));

    if (!maybe_capsule) {
        // we failed to allocate the python capsule, reraise
        return nullptr;
    }

    auto& [pycapsule, vector] = *maybe_capsule;

    auto descr = new_dtype<T>();
    if (!descr) {
        return nullptr;
    }

    auto arr = scoped_ref(PyArray_NewFromDescr(
        &PyArray_Type,
        descr.get(),
        ndim,
        const_cast<npy_intp*>(reinterpret_cast<const npy_intp*>(shape.data())),
        const_cast<npy_intp*>(reinterpret_cast<const npy_intp*>(strides.data())),
        vector.data(),
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
scoped_ref<PyObject> move_to_numpy_array(std::vector<T>&& values) {
    return move_to_numpy_array<T, 1>(std::move(values), {values.size()}, {sizeof(T)});
}
}  // namespace py
