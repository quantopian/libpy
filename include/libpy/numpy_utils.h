#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <ostream>
#include <type_traits>
#include <vector>

#include "libpy/any.h"
#include "libpy/automethod.h"
#include "libpy/borrowed_ref.h"
#include "libpy/buffer.h"
#include "libpy/char_sequence.h"
#include "libpy/datetime64.h"
#include "libpy/detail/numpy.h"
#include "libpy/exception.h"
#include "libpy/ndarray_view.h"
#include "libpy/object_map_key.h"
#include "libpy/scoped_ref.h"
#include "libpy/to_object.h"

/** Helper macro for IMPORT_ARRAY_MODULE_SCOPE below. This is needed because
    import_array() contains a return statement, and in py2 it returns void, whereas in py3
    it returns a null pointer. We want a version that consistently doesn't return, so we
    wrap the import_array call in a lambda and call the lambda.
*/
#if PY_MAJOR_VERSION == 2
#define DO_IMPORT_ARRAY() []() -> void { import_array(); }()
#else
#define DO_IMPORT_ARRAY()                                                                \
    []() -> std::nullptr_t {                                                             \
        import_array();                                                                  \
        return nullptr;                                                                  \
    }()
#endif

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
                DO_IMPORT_ARRAY();                                                       \
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
constexpr char buffer_format<py::py_bool> = '?';

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

template<>
struct new_dtype<std::int8_t> : new_dtype_from_typecode<NPY_INT8> {};

template<>
struct new_dtype<std::int16_t> : new_dtype_from_typecode<NPY_INT16> {};

template<>
struct new_dtype<std::int32_t> : new_dtype_from_typecode<NPY_INT32> {};

template<>
struct new_dtype<std::int64_t> : new_dtype_from_typecode<NPY_INT64> {};

template<>
struct new_dtype<std::uint8_t> : new_dtype_from_typecode<NPY_UINT8> {};

template<>
struct new_dtype<std::uint16_t> : new_dtype_from_typecode<NPY_UINT16> {};

template<>
struct new_dtype<std::uint32_t> : new_dtype_from_typecode<NPY_UINT32> {};

template<>
struct new_dtype<std::uint64_t> : new_dtype_from_typecode<NPY_UINT64> {};

template<>
struct new_dtype<float> : new_dtype_from_typecode<NPY_FLOAT32> {};

template<>
struct new_dtype<double> : new_dtype_from_typecode<NPY_FLOAT64> {};

template<>
struct new_dtype<PyObject*> : new_dtype_from_typecode<NPY_OBJECT> {};

template<typename T>
struct new_dtype<py::borrowed_ref<T>> : new_dtype_from_typecode<NPY_OBJECT> {};

template<typename T>
struct new_dtype<scoped_ref<T>> : new_dtype<PyObject*> {};

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
    static py::scoped_ref<> f(py_bool v) {
        return py::scoped_ref{PyBool_FromLong(v.value)};
    }
};

/** Convert a datetime64 in to a numpy array scalar.
 */
template<typename unit>
struct to_object<datetime64<unit>> {
    static py::scoped_ref<> f(const datetime64<unit>& dt) {
        auto descr = py::new_dtype<datetime64<unit>>();
        if (!descr) {
            return nullptr;
        }
        std::int64_t as_int = static_cast<std::int64_t>(dt);
        return py::scoped_ref{PyArray_Scalar(&as_int, descr.get(), nullptr)};
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
    static ndarray_view<T, ndim> f(py::borrowed_ref<> ob) {
        if (!PyArray_Check(ob.get())) {
            throw invalid_conversion::make<ndarray_view<T, ndim>>(ob);
        }

        auto array = reinterpret_cast<PyArrayObject*>(ob.get());

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
            if (!(std::is_same_v<T, py::any_cref> || PyArray_ISWRITEABLE(array))) {
                throw exception(PyExc_TypeError,
                                "cannot take a mutable view over an immutable array");
            }
            any_vtable vtable = dtype_to_vtable(given_dtype);
            using view_type = ndarray_view<T, ndim>;
            return view_type(reinterpret_cast<typename view_type::buffer_type>(
                                 PyArray_BYTES(array)),
                             shape,
                             strides,
                             vtable);
        }
        else {
            if (!(std::is_const_v<T> || PyArray_ISWRITEABLE(array))) {
                throw exception(PyExc_TypeError,
                                "cannot take a mutable view over an immutable array");
            }
            // note: This is a "constexpr else", removing and unindenting this
            // else block would have semantic meaning and be incorrect. This
            // branch is only expanded when the above test is false; if the
            // "else" is removed, it will always be expanded.

            auto expected_dtype = py::new_dtype<std::remove_cv_t<T>>();
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

            return ndarray_view<T, ndim>(reinterpret_cast<T*>(PyArray_BYTES(array)),
                                         shape,
                                         strides);
        }
    }
};

template<typename D>
struct from_object<datetime64<D>> {
    static datetime64<D> f(py::borrowed_ref<> ob) {
        if (!PyArray_CheckScalar(ob.get())) {
            throw invalid_conversion::make<datetime64<D>>(ob);
        }

        py::scoped_ref array(
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
                                                  std::move(descr),
                                                  {values.size()},
                                                  {sizeof(T)});
}

inline scoped_ref<> move_to_numpy_array(py::any_vector&& values) {

    auto descr = py::dispatch::vtable_to_dtype(values.vtable());
    if (!descr) {
        return nullptr;
    }
    return move_to_numpy_array<py::any_vector, 1>(std::move(values),
                                                  std::move(descr),
                                                  {values.size()},
                                                  {static_cast<std::int64_t>(
                                                      values.vtable().size())});
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
