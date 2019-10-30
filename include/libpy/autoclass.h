#pragma once

#include <forward_list>
#include <sstream>
#include <type_traits>
#include <typeindex>
#include <vector>

#include "libpy/automethod.h"
#include "libpy/build_tuple.h"
#include "libpy/demangle.h"
#include "libpy/detail/autoclass_cache.h"
#include "libpy/detail/autoclass_object.h"
#include "libpy/detail/autoclass_py2.h"
#include "libpy/detail/python.h"
#include "libpy/meta.h"
#include "libpy/scope_guard.h"

namespace py {
/** Zero the memory that is not part of the base PyObject.

    @param b The object to zero the non-PyObject component.
 */
template<typename T>
constexpr void zero_non_pyobject_base(T* b) {
    static_assert(sizeof(T) >= sizeof(PyObject));
    static_assert(std::is_pod_v<T>,
                  "if using zero_non_pyobject_base, base type must a POD type");

    std::byte* as_bytes = reinterpret_cast<std::byte*>(b);
    std::memset(as_bytes + sizeof(PyObject), 0, sizeof(T) - sizeof(PyObject));
}

namespace detail {
template<typename T>
constexpr void nop_clear_base(T*) {}
}  // namespace detail

/** A factory for generating a Python class that wraps instances of a C++ type.

    @tparam T The C++ type to be wrapped.
    @tparam base The static base type of the Python instances created.
    @tparam initialize_base A function used to initialize the Python base object.
    @tparam clear_base A function used to clear the Python base object fields.

    ### Usage

    To create a new Python type for an object that wraps a C++ type, `my_type`,
    you can write the following:

    ```
    py::scoped_ref<PyTypeObject> t = py::autoclass<my_type>("modname.PythonName").type();
    ```

    The resulting type will have a `__name__` of "PythonName", and a
    `__module__` of "modname".

    By default, types created with `autoclass` expose no functionality to
    Python, not even a constructor. To add a constructor to your python object,
    invoke the `.new_` method of the `autoclass`, templated with the C++
    signature of the constructor. For example, to create a constructor that
    accepts an int and a double, we would write:

    ```
    py::scoped_ref<PyTypeObject> t = py::autoclass<my_type>("modname.PythonName")
                                         .new_<int, double>()
                                         .type();
    ```

    Only one constructor overload may be exposed to Python because Python
    doesn't support function overloading. If a constructor is not provided,
    instances can only be constructed from C++ using
    `py::autoclass<my_type>::construct(Args&&...)`.

    To expose methods of your C++ type to Python, call `.def`, templated on the
    address of the C++ method you want to expose, and pass the name of the
    Python method to generate. For example, to expose a C++ method called `foo`
    as a Python method named `bar`, we would write:

    ```
    py::scoped_ref<PyTypeObject> t = py::autoclass<my_type>("modname.PythonName")
                                         .new_<int, double>()
                                         .def<&my_type::foo>("bar")
                                         .type();
    ```

    ### Subclassing Existing Python Types

    Python instances are composed of two parts:

    - A static C or C++ type that defines the layout of instances in memory. A
    - runtime value of type `PyTypeObject`, which represents the Python type of
      the object.

    An object's static type must contain at least the state defined by the base
    `PyObject` struct, which includes a `PyTypeObject*` that points to the
    object's python type. In general, there is a one-to-many relationship
    between static object layouts and Python types (i.e. `PyTypeObject*`s).
    Objects of a given Python type always have the same layout, but multiple
    objects of different Python types may have the same layout if they have the
    same state. For example, the standard Python exceptions all have the same
    layout (defined by the `PyBaseExceptionObject` struct), but they have
    different Python types so that users can catch specific exceptions types.

    By default, Python objects defined with `autoclass<T>.type()` will have a
    layout that looks like:

    ```
    class autoclass_object : public PyObject {
        T value;
    };
    ```

    The `PyTypeObject` returned by `autoclass::type` will be a (Python) subclass
    of Python's `object` type.

    In rare cases, it may be useful to use autoclass to define a type that
    subclasses from a Python type other than `object`. To support this use-case,
    `autoclass` allows you to provide a few optional arguments:

    - `base`: the static instance type to subclass instead of `PyObject`.
    - `initialize_base`: a function which initializes the non-`PyObject` fields
       of the `base` struct.
    - `clear_base`: a function which tears down the non-`PyObject` fields of the
      `base` struct.

    By default, the non-`PyObject` fields of `base` will just be zeroed and no
    cleanup is performed. Subclassing existing Python types is an advanced
    autoclass feature and is not guaranteed to be safe in all configurations.
    Please use this feature with care.
 */
template<typename T,
         typename base = PyObject,
         auto initialize_base = zero_non_pyobject_base<base>,
         auto clear_base = detail::nop_clear_base<base>>
class autoclass {
private:
    /** Marker that statically asserts that the unsafe API is enabled.

        Call this function first in any top-level function that is part of the unsafe API.
     */
    template<typename U = void>
    static void unsafe_api() {
        // We need to put this in a template-dependent scope so that the static_assert
        // only fails if you call an unsafe API function. `is_samve_V<U, U>` will always
        // return true, but it becomes a template-dependent scope because it depends on
        // `U`. If `!LIBPY_AUTOCLASS_UNSAFE_API` we put a `!` in front of the result to
        // negate it.
        constexpr bool is_valid =
#if !LIBPY_AUTOCLASS_UNSAFE_API
            !
#endif
            std::is_same_v<U, U>;

        static_assert(is_valid,
                      "This function is only available in the unsafe API. This may be "
                      "enabled by defining the LIBPY_AUTOCLASS_UNSAFE_API macro.");
    }

public:
    using object = detail::autoclass_object<T, base>;

    template<typename U>
    static T& unbox(const U& ptr) {
        return object::unbox(ptr);
    }

private:
    std::vector<PyType_Slot> m_slots;
    std::unique_ptr<detail::autoclass_storage> m_storage;
    py::scoped_ref<PyTypeObject> m_type;
    PyType_Spec m_spec;
    py::scoped_ref<PyTypeObject> m_py_basetype;

    /** Check if this type uses the `Py_TPFLAGS_HAVE_GC`, which requires that we implement
        at least `Py_tp_traverse`, and will use `PyObject_GC_New` and `PyObject_GC_Del`.
     */
    bool have_gc() const {
        return m_spec.flags & Py_TPFLAGS_HAVE_GC;
    }

    /** Helper for adapting a member function of `T` into a Python method.
     */
    template<typename F, auto f>
    struct member_function;

    template<auto impl, typename R, typename... Args>
    struct pointer_to_member_function_base {
        static R f(PyObject* self, Args... args) {
            return (unbox(self).*impl)(std::forward<Args>(args)...);
        }
    };

    template<auto impl, typename R, typename... Args>
    struct free_function_base {
        static R f(PyObject* self, Args... args) {
            return impl(unbox(self), std::forward<Args>(args)...);
        }
    };

    // dispatch for non-const member function
    template<typename C, typename R, typename... Args, auto impl>
    struct member_function<R (C::*)(Args...), impl>
        : public pointer_to_member_function_base<impl, R, Args...> {};

    // dispatch for non-const noexcept member function
    template<typename C, typename R, typename... Args, auto impl>
    struct member_function<R (C::*)(Args...) noexcept, impl>
        : public pointer_to_member_function_base<impl, R, Args...> {};

    // dispatch for const member function
    template<typename C, typename R, typename... Args, auto impl>
    struct member_function<R (C::*)(Args...) const, impl>
        : public pointer_to_member_function_base<impl, R, Args...> {};

    // dispatch for const noexcept member function
    template<typename C, typename R, typename... Args, auto impl>
    struct member_function<R (C::*)(Args...) const noexcept, impl>
        : public pointer_to_member_function_base<impl, R, Args...> {};

    // dispatch for free function that accepts as a first argument `T`
    template<typename R, typename... Args, auto impl>
    struct member_function<R(T, Args...), impl>
        : public free_function_base<impl, R, Args...> {};

    // dispatch for free function that accepts as a first argument `T&`
    template<typename R, typename... Args, auto impl>
    struct member_function<R(T&, Args...), impl>
        : public free_function_base<impl, R, Args...> {};

    // dispatch for free function that accepts as a first argument `const T&`
    template<typename R, typename... Args, auto impl>
    struct member_function<R(const T&, Args...), impl>
        : public free_function_base<impl, R, Args...> {};

    // dispatch for a noexcept free function that accepts as a first argument `T`
    template<typename R, typename... Args, auto impl>
    struct member_function<R(T, Args...) noexcept, impl>
        : public free_function_base<impl, R, Args...> {};

    // dispatch for noexcept free function that accepts as a first argument `T&`
    template<typename R, typename... Args, auto impl>
    struct member_function<R(T&, Args...) noexcept, impl>
        : public free_function_base<impl, R, Args...> {};

    // dispatch for a noexcept free function that accepts as a first argument `const T&`
    template<typename R, typename... Args, auto impl>
    struct member_function<R(const T&, Args...) noexcept, impl>
        : public free_function_base<impl, R, Args...> {};

    /** Assert that `m_storage->type` isn't yet initialized. Many operations like adding
       methods will not have any affect after the type is initialized.

        @param msg The error message to forward to the `ValueError` thrown if this
        assertion is violated.
     */
    void require_uninitialized(const char* msg) {
        if (m_type) {
            throw py::exception(PyExc_ValueError, msg);
        }
    }

protected:
    template<typename, typename, auto, auto>
    friend class autoclass;

    /** Add a slot to the spec.

        @param slot_id The id of the slot to add.
        @param to_add The value of the slot.
     */
    template<typename U>
    autoclass& add_slot(int slot_id, U* to_add) {
        m_slots.push_back(PyType_Slot{slot_id, reinterpret_cast<void*>(to_add)});
        return *this;
    }

private:
    /** Mark the end of slots.
     */
    void finalize_slots() {
        m_slots.push_back(PyType_Slot{0, nullptr});
    }

public:
    /** Look up an already created type.

        @return The already created type, or `nullptr` if the type wasn't yet created.
     */
    static py::scoped_ref<PyTypeObject> lookup_type() {
        auto type_search = detail::autoclass_type_cache.get().find(typeid(T));
        if (type_search != detail::autoclass_type_cache.get().end()) {
            PyTypeObject* type = type_search->second->type;
            Py_INCREF(type);
            return py::scoped_ref(type);
        }
        return nullptr;
    }

private:
    /** Construct a new Python object that wraps a `T` without boxing/unboxing the
        constructor arguments.

        @param args The arguments to forward to the C++ type's constructor.
        @return A pair of the Python type for `T` and a new reference to a
                Python wrapped version of `T`, or nullptr on failure.
    */
    template<typename... Args>
    static std::tuple<py::scoped_ref<PyTypeObject>, py::scoped_ref<>>
    construct_with_type(Args&&... args) {
        auto cls = lookup_type();
        if (!cls) {
            py::raise(PyExc_RuntimeError)
                << "C++ type " << util::type_name<T>().get()
                << " does not have a corresponding Python type yet";
            return {nullptr, nullptr};
        }

        PyObject* out;
        if (cls.get()->tp_flags & Py_TPFLAGS_HAVE_GC) {
            out = constructor_new_impl<true>(cls.get(), std::forward<Args>(args)...);
        }
        else {
            out = constructor_new_impl<false>(cls.get(), std::forward<Args>(args)...);
        }

        return {std::move(cls), py::scoped_ref(out)};
    }

public:
    /** Construct a new Python object that wraps a `T` without boxing/unboxing the
        constructor arguments.

        @param args The arguments to forward to the C++ type's constructor.
        @return A new reference to a Python wrapped version of `T`, or nullptr on failure.
     */
    template<typename... Args>
    static py::scoped_ref<> construct(Args&&... args) {
        auto [cls, ob] = construct_with_type(std::forward<Args>(args)...);
        return ob;
    }

    /** Raise a Python exception with a wrapped version of `T` without
        boxing/unboxing the constructor arguments.

        @param args The arguments to forward to the C++ type's constructor.
        @return `nullptr`.
     */
    template<typename... Args>
    static std::nullptr_t raise(Args&&... args) {
        auto [cls, ob] = construct_with_type(std::forward<Args>(args)...);
        if (!ob) {
            return nullptr;
        }
        PyErr_SetObject(static_cast<PyObject*>(cls), ob.get());
        return nullptr;
    }

private:
    /** Get the `tp_flags` from the user provided extra flags and Python base type.

        @param extra_flags any extra flags the user explicitly provided.
        @param base_type A potentially null pointer to the Python base type.
        @return The `tp_flags` to use for this autoclass generated type.
     */
    static unsigned int flags(int extra_flags, PyTypeObject* base_type) {
        unsigned int out = detail::autoclass_base_flags;
        out |= extra_flags;
        if (base_type && base_type->tp_flags & Py_TPFLAGS_HAVE_GC) {
            out |= Py_TPFLAGS_HAVE_GC;
        }
        return out;
    }

public:
    autoclass(std::string name = util::type_name<T>().get(),
              int extra_flags = 0,
              PyTypeObject* base_type = nullptr)
        : m_storage(std::make_unique<detail::autoclass_storage>(std::move(name))),
          m_type(nullptr),
          m_spec({m_storage->strings.front().data(),
                  static_cast<int>(sizeof(object)),
                  0,
                  flags(extra_flags, base_type),
                  nullptr}),
          m_py_basetype(py::scoped_ref<PyTypeObject>::xnew_reference(base_type)) {
        if (base_type) {
            // Check to make sure that the static base type is not obviously
            // wrong. This check does not ensure that the static base type is
            // compatible with `base_type`, but if this check fails, then they
            // are certainly incompatible.
            if (base_type->tp_basicsize != sizeof(base)) {
                throw util::formatted_error<std::invalid_argument>(
                    "Python base type ",
                    base_type->tp_name,
                    " instances do not match the size of `base` template param ",
                    util::type_name<base>().get(),
                    ": ",
                    base_type->tp_basicsize,
                    " != ",
                    sizeof(base));
            }

            if (base_type->tp_itemsize) {
                throw std::invalid_argument{
                    "Python base type is a varobject, which is unsupported as an "
                    "autoclass type's base type"};
            }

            add_slot(Py_tp_base, base_type);
        }
        auto type_search = detail::autoclass_type_cache.get().find(typeid(T));
        if (type_search != detail::autoclass_type_cache.get().end()) {
            throw std::runtime_error{"type was already created"};
        }

        void (*py_dealloc)(PyObject*);
        if (have_gc()) {
            py_dealloc = [](PyObject* self) {
                PyObject_GC_UnTrack(self);
                clear_base(reinterpret_cast<base*>(self));
                unbox(self).~T();
                dealloc(true, self);
            };
        }
        else {
            py_dealloc = [](PyObject* self) {
                if (!std::is_same_v<base, PyObject>) {
                    if (auto clear = self->ob_type->tp_base->tp_clear; clear) {
                        clear(self);
                    }
                }
                unbox(self).~T();
                dealloc(false, self);
            };
        }
        add_slot(Py_tp_dealloc, py_dealloc);
    }

    // Delete the copy constructor, the intermediate string data points into
    // storage that is managed by the type until `.type()` is called.
    // Also, don't try to create 2 versions of the same type.
    autoclass(const autoclass&) = delete;
    autoclass(autoclass&&) = default;
    autoclass& operator=(autoclass&&) = default;

    /** Add a `tp_traverse` field to this type. This is only allowed, but required if
        `extra_flags & Py_TPFLAGS_HAVE_GC`.

        @tparam impl The implementation of the traverse function. This should either be an
                     `int(T&, visitproc, void*)` or `int (T::*)(visitproc, void*)`.
     */
    template<auto impl>
    autoclass& traverse() {
        require_uninitialized(
            "cannot add traverse method after the class has been created");

        if (!have_gc()) {
            throw py::exception(PyExc_TypeError,
                                "cannot add a traverse method without passing "
                                "Py_TPFLAGS_HAVE_GC to extra_flags");
        }

        traverseproc p = [](PyObject* self, visitproc visit, void* arg) -> int {
            try {
                return std::invoke(impl, unbox(self), visit, arg);
            }
            catch (const std::exception& e) {
                py::raise_from_cxx_exception(e);
                return -1;
            }
        };
        return add_slot(Py_tp_traverse, p);
    }

    /** Add a `tp_clear` field to this type. This is only allowed if
        `extra_flags & Py_TPFLAGS_HAVE_GC`.

        @tparam impl The implementation of the clear function. This should either be an
                     `int(T&)` or `int (T::*)()`.
     */
    template<auto impl>
    autoclass& clear() {
        require_uninitialized("cannot add clear method after the class has been created");

        if (!have_gc()) {
            throw py::exception(PyExc_TypeError,
                                "cannot add a clear method without passing "
                                "Py_TPFLAGS_HAVE_GC to extra_flags");
        }

        inquiry p = [](PyObject* self) -> int {
            try {
                return std::invoke(impl, unbox(self));
            }
            catch (const std::exception& e) {
                py::raise_from_cxx_exception(e);
                return -1;
            }
        };
        return add_slot(Py_tp_clear, p);
    }

    /** Add a docstring to this class.

        @param doc
        @return *this.
     */
    autoclass& doc(std::string doc) {
        require_uninitialized("cannot add docstring after the class has been created");

        std::string& copied_doc = m_storage->strings.emplace_front(std::move(doc));
        return add_slot(Py_tp_doc, copied_doc.data());
    }

private:
    static object* alloc(bool have_gc, PyTypeObject* cls) {
        if (have_gc) {
            return PyObject_GC_New(object, cls);
        }
        return PyObject_New(object, cls);
    }

    template<typename b = base, typename = std::enable_if_t<!std::is_same_v<b, PyObject>>>
    static void dealloc(bool have_gc, base* ob) {
        dealloc(have_gc, reinterpret_cast<PyObject*>(ob));
    }

    static void dealloc(bool have_gc, PyObject* ob) {
        if (have_gc) {
            PyObject_GC_Del(ob);
        }
        else {
            PyObject_Del(ob);
        }
    }

    /** Helper for adapting a function which constructs a `T` into a Python `__new__`
        implementation.
    */
    template<bool have_gc, typename F, auto impl>
    struct free_func_new_impl;

    template<bool have_gc, typename R, typename... Args, auto impl>
    struct free_func_new_impl<have_gc, R(Args...), impl> {
        static PyObject* f(PyTypeObject* cls, Args... args) {
            return constructor_new_impl<have_gc,
                                        std::invoke_result_t<decltype(impl), Args...>>(
                cls, impl(args...));
        }
    };

    template<bool have_gc, typename R, typename... Args, auto impl>
    struct free_func_new_impl<have_gc, R (*)(Args...), impl> {
        static PyObject* f(PyTypeObject* cls, Args... args) {
            return constructor_new_impl<have_gc,
                                        std::invoke_result_t<decltype(impl), Args...>>(
                cls, impl(args...));
        }
    };

    /** Function which will be used to expose one of `T`'s constructors as the `__new__`.
     */
    template<bool have_gc, typename... ConstructorArgs>
    static PyObject* constructor_new_impl(PyTypeObject* cls, ConstructorArgs... args) {
        object* self = alloc(have_gc, cls);
        if (!self) {
            return nullptr;
        }

        try {
            initialize_base(self);
        }
        catch (...) {
            dealloc(have_gc, self);
        }

        try {
            new (&unbox(self)) T(std::forward<ConstructorArgs>(args)...);
        }
        catch (...) {
            clear_base(self);
            dealloc(have_gc, self);
            throw;
        }

        if (have_gc) {
            PyObject_GC_Track(self);
        }
        return reinterpret_cast<PyObject*>(self);
    }

public:
    /** Add a `__new__` function to this class.

        @tparam impl A function which returns a value which can be used to construct a
                `T`.
        @return *this.
     */
    template<auto impl>
    autoclass& new_() {
        require_uninitialized("cannot add __new__ after the class has been created");

        PyObject* (*new_)(PyTypeObject*, PyObject*, PyObject*);
        if (m_spec.flags & Py_TPFLAGS_HAVE_GC) {
            new_ = wrap_new<free_func_new_impl<true, decltype(impl), impl>::f>;
        }
        else {
            new_ = wrap_new<free_func_new_impl<false, decltype(impl), impl>::f>;
        }
        return add_slot(Py_tp_new, new_);
    }

    /** Add a `__new__` function to this class by adapting one of the constructors of `T`.

        @tparam ConstructorArgs The C++ signature of the constructor to use.
        @return *this.
     */
    template<typename... ConstructorArgs>
    autoclass& new_() {
        require_uninitialized("cannot add __new__ after the class has been created");

        PyObject* (*new_)(PyTypeObject*, PyObject*, PyObject*);
        if (have_gc()) {
            new_ = wrap_new<constructor_new_impl<true, ConstructorArgs...>>;
        }
        else {
            new_ = wrap_new<constructor_new_impl<false, ConstructorArgs...>>;
        }
        return add_slot(Py_tp_new, new_);
    }

private:
    template<typename U>
    lenfunc get_length_func() {
        return [](PyObject* self) -> Py_ssize_t {
            try {
                return static_cast<Py_ssize_t>(unbox(self).size());
            }
            catch (std::exception& e) {
                raise_from_cxx_exception(e);
                return -1;
            }
        };
    }

    template<typename U,
             typename =
                 std::void_t<decltype(static_cast<Py_ssize_t>(std::declval<U>().size()))>>
    lenfunc maybe_get_length_func(int) {
        return get_length_func<U>();
    }

    template<typename>
    lenfunc maybe_get_length_func(long) {
        return nullptr;
    }

public:
    /** Add a `__len__` method from `T::size()`
     */
    autoclass& len() {
        // this isn't really unsafe, but it is just dumb without `iter()` or `mapping()`.
        unsafe_api();
        require_uninitialized("cannot add size method after the class has been created");
        return add_slot(Py_mp_length, get_length_func<T>());
    }

private:
    /** Template to filter `RHS` down to only the valid types which may appear on the RHS
        of some binary operator.

        @tparam F A template that takes two types and provides a
                  `static constexpr bool value` member which indicates whether `T op RHS`
                  is valid. This should be obtained from:
                  `test_binop<op>::template check`.
        @tparam RHS The candidate RHS types as a `std::tuple`.
    */
    template<template<typename, typename> typename F, typename RHS>
    struct valid_rhs_types;

    // Partial specialization for non-empty tuples.
    template<template<typename, typename> typename F, typename Head, typename... Tail>
    struct valid_rhs_types<F, std::tuple<Head, Tail...>> {
    private:
        using rest = std::tuple<Tail...>;

    public:
        // put `Head` in the output tuple if there is a path to `Head` from `PyObject*`.
        using type = std::conditional_t<
            (std::is_same_v<Head, T> || py::has_from_object<Head>) &&F<T, Head>::value,
            meta::type_cat<std::tuple<Head>, typename valid_rhs_types<F, rest>::type>,
            typename valid_rhs_types<F, rest>::type>;
    };

    // Base-case: empty list of types
    template<template<typename, typename> typename F>
    struct valid_rhs_types<F, std::tuple<>> {
        using type = std::tuple<>;
    };

    template<typename F, typename LHS, typename RHS>
    struct search_binop_implementations_helper;

    template<typename F, typename LHS, typename Head, typename... Tail>
    struct search_binop_implementations_helper<F, LHS, std::tuple<Head, Tail...>> {
        static PyObject* f(F&& f, LHS& lhs, PyObject* rhs) {
            bool rethrow = false;
            try {
                const Head& unboxed_rhs = py::from_object<const Head&>(rhs);
                rethrow = true;
                return py::to_object(f(lhs, unboxed_rhs)).escape();
            }
            catch (const py::invalid_conversion&) {
                if (rethrow) {
                    throw;
                }
                PyErr_Clear();
                return search_binop_implementations_helper<F, LHS, std::tuple<Tail...>>::
                    f(std::forward<F>(f), lhs, rhs);
            }
        }
    };

    template<typename F, typename LHS>
    struct search_binop_implementations_helper<F, LHS, std::tuple<>> {
        static PyObject* f(F&&, LHS&, PyObject*) {
            Py_RETURN_NOTIMPLEMENTED;
        }
    };

    /** Check all possible implementations of `f(lhs, rhs)`, where `rhs` is a `PyObject*`
        and needs to be converted to a C++ value.

        If `other` cannot be converted to any type in `RHS`, return `NotImplemented`.

        @tparam RHS A `std::tuple` of all of the possible C++ types for `RHS` to be
                    converted to.
        @tparam F The binary function to search.
        @tparam LHS Set to `T`, but lazily to be SFINAE friendly.
        @param f The binary function to search.
        @param lhs The lhs argument to `f`.
        @param rhs The rhs argument to `f` as a `PyObject*`.
        @return The Python result of `f(lhs, rhs)`.

        @note RHS appears first in the template argument list because it must be
              explicitly provided, where `F` and `LHS` are meant to be inferred from the
              parameters.
     */
    template<typename RHS, typename F, typename LHS>
    static PyObject* search_binop_implementations(F&& f, LHS& lhs, PyObject* rhs) {
        return search_binop_implementations_helper<F, LHS, RHS>::f(std::forward<F>(f),
                                                                   lhs,
                                                                   rhs);
    }

    /** Curried metafunction to check if `op` will be callable with values of type `L`
        and `R`.
     */
    template<typename op>
    struct test_binop {
        /** The binary metafunction which checks if `op` is callable with values of type
            `L` and `R`.
         */
        template<typename L, typename R>
        struct check {
        private:
            template<typename LHS, typename RHS>
            static decltype(op{}(std::declval<LHS>(), std::declval<RHS>()),
                            std::true_type{})
            valid(int);

            template<typename, typename>
            static std::false_type valid(long);

        public:
            static constexpr bool value =
                std::is_same_v<decltype(valid<L, R>(0)), std::true_type>;
        };
    };

    template<typename U,
             typename op,
             typename RHS,
             // enabled iff there is at least one value RHS type
             typename = std::enable_if_t<!std::is_same_v<
                 std::tuple<>,
                 typename valid_rhs_types<test_binop<op>::template check, RHS>::type>>>
    static constexpr auto get_binop_func_impl(int) {
        using valid_rhs =
            typename valid_rhs_types<test_binop<op>::template check, RHS>::type;
        return [](PyObject* self, PyObject* other) -> PyObject* {
            try {
                if (meta::element_of<T, valid_rhs> && Py_TYPE(self) == Py_TYPE(other)) {
                    // rhs is an exact match of `T`, and `T` is a valid RHS
                    return py::to_object(op{}(unbox(self), unbox(other))).escape();
                }
                else {
                    // Try to convert `other` into each C++ type in `rhs_without_T`
                    // (linear search).  If `other` is successfully converted, return
                    // `self + other`. If no conversion matches, return `NotImplemented`.
                    using rhs_without_T = meta::set_diff<valid_rhs, std::tuple<T>>;
                    return search_binop_implementations<rhs_without_T>(op{},
                                                                       unbox(self),
                                                                       other);
                }
            }
            catch (const std::exception& e) {
                return raise_from_cxx_exception(e);
            }
        };
    }

    template<typename, typename, typename>
    static constexpr binaryfunc get_binop_func_impl(long) {
        return nullptr;
    }

    template<typename LHS, typename op, typename RHS>
    static constexpr binaryfunc arith_func = get_binop_func_impl<LHS, op, RHS>(0);

    /** Look up a binary function as a comparison implementation to be used in
        richcompare.
     */
    template<typename LHS, typename op, typename RHS>
    static constexpr auto get_cmp_func() {
        auto out = get_binop_func_impl<LHS, op, RHS>(0);
        if constexpr (std::is_same_v<decltype(out), binaryfunc>) {
            return [](PyObject*, PyObject*) { Py_RETURN_NOTIMPLEMENTED; };
        }
        else {
            return out;
        }
    }

    template<typename LHS, typename op, typename RHS>
    static constexpr auto cmp_func = get_cmp_func<LHS, op, RHS>();

public:
    /** Add all of the number methods by inferring them from `T`'s
        binary operators.

        @tparam BinOpRHSTypes The types to consider as valid RHS types for arithmetic.
     */
    template<typename... BinOpRHSTypes>
    autoclass& arithmetic() {
        require_uninitialized(
            "cannot add arithmetic methods after the class has been created");

        using RHS = std::tuple<BinOpRHSTypes...>;

        add_slot(Py_nb_add, arith_func<T, meta::op::add, RHS>);
        add_slot(Py_nb_subtract, arith_func<T, meta::op::sub, RHS>);
        add_slot(Py_nb_multiply, arith_func<T, meta::op::mul, RHS>);
        add_slot(Py_nb_remainder, arith_func<T, meta::op::rem, RHS>);
        add_slot(Py_nb_floor_divide, arith_func<T, meta::op::div, RHS>);
        add_slot(Py_nb_lshift, arith_func<T, meta::op::lshift, RHS>);
        add_slot(Py_nb_rshift, arith_func<T, meta::op::rshift, RHS>);
        add_slot(Py_nb_and, arith_func<T, meta::op::and_, RHS>);
        add_slot(Py_nb_xor, arith_func<T, meta::op::xor_, RHS>);
        add_slot(Py_nb_or, arith_func<T, meta::op::or_, RHS>);
        return *this;
    }

private:
    template<typename RHS>
    static PyObject* richcompare_impl(PyObject* self, PyObject* other, int cmp) {
        try {
            switch (cmp) {
            case Py_LT:
                return cmp_func<T, meta::op::lt, RHS>(self, other);
            case Py_LE:
                return cmp_func<T, meta::op::le, RHS>(self, other);
            case Py_EQ:
                return cmp_func<T, meta::op::eq, RHS>(self, other);
            case Py_GE:
                return cmp_func<T, meta::op::ge, RHS>(self, other);
            case Py_GT:
                return cmp_func<T, meta::op::gt, RHS>(self, other);
            case Py_NE:
                return cmp_func<T, meta::op::ne, RHS>(self, other);
            default:
                py::raise(PyExc_SystemError) << "invalid richcompare op: " << cmp;
                return nullptr;
            }
        }
        catch (const std::exception& e) {
            return raise_from_cxx_exception(e);
        }
    }

public:
    /** Add all of the comparisons methods by inferring them from `T`'s operators.

        @tparam BinOpRHSTypes The types to consider as valid RHS types for the
                comparisons.
    */
    template<typename... BinOpRHSTypes>
    autoclass& comparisons() {
        static_assert(sizeof...(BinOpRHSTypes) > 0,
                      "comparisons() requires at least one BinOpRHSTypes");

        require_uninitialized(
            "cannot add comparison methods after the class has been created");

        richcmpfunc impl = &richcompare_impl<std::tuple<BinOpRHSTypes...>>;
        return add_slot(Py_tp_richcompare, impl);
    }

private:
    template<typename U,
             typename op,
             typename = std::void_t<decltype(op{}(std::declval<U>()))>>
    unaryfunc get_unop_func(int) {
        return [](PyObject* self) -> PyObject* {
            try {
                return py::to_object(op{}(unbox(self))).escape();
            }
            catch (const std::exception& e) {
                return raise_from_cxx_exception(e);
            }
        };
    }

    template<typename, typename>
    static constexpr unaryfunc get_unop_func(long) {
        return nullptr;
    }

public:
    /** Add unary operator methods `__neg__` and `__pos__`, and `__inv__` from the C++
        `operator-`, `operator+`, and `operator~` respectively.  interface.

     */
    autoclass& unary() {
        require_uninitialized(
            "cannot add unary operator methods after the class has been created");

        add_slot(Py_nb_negative, get_unop_func<T, meta::op::neg>(0));
        add_slot(Py_nb_positive, get_unop_func<T, meta::op::pos>(0));
        add_slot(Py_nb_invert, get_unop_func<T, meta::op::inv>(0));
        return *this;
    }

private:
    template<typename U,
             typename type,
             typename = std::void_t<decltype(static_cast<type>(std::declval<U>()))>>
    unaryfunc get_convert_func(int) {
        return [](PyObject* self) -> PyObject* {
            try {
                return py::to_object(static_cast<type>(unbox(self))).escape();
            }
            catch (const std::exception& e) {
                return raise_from_cxx_exception(e);
            }
        };
    }

    template<typename, typename>
    unaryfunc get_convert_func(long) {
        return nullptr;
    }

    template<typename U,
             typename = std::void_t<decltype(static_cast<bool>(std::declval<U>()))>>
    inquiry get_convert_bool_func(int) {
        return [](PyObject* self) -> int {
            try {
                // `inquiry` returns an `int` because it is C, but really wants to be
                // `std::optional<bool>`. Convert to bool first (because that's what we
                // mean, not int), then convert that to int to get 0 or 1.
                return static_cast<bool>(unbox(self));
            }
            catch (const std::exception& e) {
                raise_from_cxx_exception(e);
                return -1;
            }
        };
    }

    template<typename>
    inquiry get_convert_bool_func(long) {
        return nullptr;
    }

public:
    /** Add conversions to `int`, `float`, and `bool` by delegating to
        `static_cast<U>(T)`.
     */
    autoclass& conversions() {
        require_uninitialized(
            "cannot add conversion methods after the class has been created");

        // conversions
        add_slot(Py_nb_int, get_convert_func<T, std::int64_t>(0));
        add_slot(Py_nb_float, get_convert_func<T, double>(0));
        add_slot(Py_nb_bool, get_convert_bool_func<T>(0));
        return *this;
    }

private:
    template<typename U, typename KeyType>
    binaryfunc get_getitem_func() {
        return [](PyObject* self, PyObject* key) -> PyObject* {
            try {
                return py::to_object(unbox(self)[from_object<const KeyType&>(key)])
                    .escape();
            }
            catch (std::exception& e) {
                return raise_from_cxx_exception(e);
            }
        };
    }

    template<typename U,
             typename KeyType,
             typename ValueType,
             typename = std::void_t<decltype(
                 std::declval<U>()[std::declval<KeyType>()] = std::declval<ValueType>())>>
    objobjargproc get_setitem_func(int) {
        return [](PyObject* self, PyObject* key, PyObject* value) {
            try {
                const ValueType& rhs = from_object<const ValueType&>(value);
                unbox(self)[from_object<const KeyType&>(key)] = rhs;
                return 0;
            }
            catch (std::exception& e) {
                raise_from_cxx_exception(e);
                return 1;
            }
        };
    }

    template<typename U, typename KeyType, typename ValueType>
    objobjargproc get_setitem_func(long) {
        return nullptr;
    }

public:
    /** Add mapping methods (``__setitem__`` and `__getitem__``) from the C++ `operator[]`
        interface.

        @tparam KeyType The type of the keys for this mapping.
        @tparam ValueType The type of the values. If not provided, or explicitly void,
                the `__setitem__` method will not be generated.
     */
    template<typename KeyType, typename ValueType = void>
    autoclass& mapping() {
        unsafe_api();
        require_uninitialized("cannot add mapping methods after type has been created");

        add_slot(Py_mp_subscript, get_getitem_func<T, KeyType>());
        add_slot(Py_mp_ass_subscript, get_setitem_func<T, KeyType, ValueType>(0));
        add_slot(Py_mp_length, maybe_get_length_func<T>(0));
        return *this;
    }

    /** Add a method to this class.

        @tparam impl The implementation of the method to add. If `impl` is a
                pointer to member function, it doesn't need to have the implicit
                `PyObject* self` argument, it will just be called on the boxed value of
                `self`.
        @tparam flags extra flags to indicate whether a function is a static or
                classmethod.
                See https://docs.python.org/3/c-api/structures.html#METH_CLASS
                for more info. Only METH_CLASS and METH_STATIC can be passed,
                the others are inferred from your function.
        @param name The name of the function as it will appear in Python.
        @return *this.
     */
    template<auto impl, int flags = 0>
    autoclass& def(std::string name) {
        require_uninitialized("cannot add methods after the class has been created");

        std::string& name_copy = m_storage->strings.emplace_front(std::move(name));
        if constexpr (flags & METH_STATIC) {
            m_storage->methods.emplace_back(autofunction<impl, flags>(name_copy.data()));
        }
        else {
            m_storage->methods.emplace_back(
                automethod<member_function<decltype(impl), impl>::f>(name_copy.data()));
        }
        return *this;
    }

    /** Add a method to this class.

        @tparam impl The implementation of the method to add. If `impl` is a
                pointer to member function, it doesn't need to have the implicit
                `PyObject* self` argument, it will just be called on the boxed value of
                `self`.
        @tparam flags extra flags to indicate whether a function is a static or
                classmethod.
        @param name The name of the function as it will appear in Python.
        @param doc The docstring of the function as it will appear in Python.
        @return *this.
     */
    template<auto impl, int flags = 0>
    autoclass& def(std::string name, std::string doc) {
        require_uninitialized("cannot add methods after the class has been created");

        std::string& name_copy = m_storage->strings.emplace_front(std::move(name));
        std::string& doc_copy = m_storage->strings.emplace_front(std::move(doc));
        if constexpr (flags & METH_STATIC || flags & METH_CLASS) {
            m_storage->methods.emplace_back(
                automethod<impl, flags>(name_copy.data(), doc_copy.data()));
        }
        else {
            m_storage->methods.emplace_back(
                automethod<member_function<decltype(impl), impl>::f>(name_copy.data(),
                                                                     doc_copy.data()));
        }
        return *this;
    }

private:
    template<typename U>
    getiterfunc get_iter_func() {
        // new-style ranges allow for begin and end to produce different types
        using begin_type = decltype(std::declval<T>().begin());
        using end_type = decltype(std::declval<T>().end());

        struct iter {
            scoped_ref<> iterable;
            begin_type it;
            end_type end;

            iter(const scoped_ref<>& iterable, begin_type it, end_type end)
                : iterable(iterable), it(it), end(end) {}

            int traverse(visitproc visit, void* arg) {
                Py_VISIT(iterable.get());
                return 0;
            }
        };

        auto iternext = [](PyObject* self) -> PyObject* {
            try {
                iter& unboxed = autoclass<iter>::unbox(self);
                if (unboxed.it != unboxed.end) {
                    py::scoped_ref out = py::to_object(*unboxed.it);
                    ++unboxed.it;
                    return std::move(out).escape();
                }
                return nullptr;
            }
            catch (const std::exception& e) {
                return raise_from_cxx_exception(e);
            }
        };

        std::string iter_name(m_spec.name);
        iter_name += "::iterator";

        // create the iterator class and put it in the cache
        if (!autoclass<iter>(std::move(iter_name), Py_TPFLAGS_HAVE_GC)
                 .add_slot(Py_tp_iternext, static_cast<iternextfunc>(iternext))
                 .add_slot(Py_tp_iter, &PyObject_SelfIter)
                 .template traverse<&iter::traverse>()
                 .type()) {
            throw py::exception{};
        }

        return [](PyObject* self) -> PyObject* {
            try {
                py::scoped_ref<PyTypeObject> cls = autoclass<iter>::lookup_type();
                if (!cls) {
                    py::raise(PyExc_RuntimeError)
                        << "no iterator type found for " << util::type_name<T>().get();
                    return nullptr;
                }

                Py_INCREF(self);
                py::scoped_ref self_ref(self);
                return autoclass<iter>::template constructor_new_impl<true>(
                    cls.get(), self_ref, unbox(self).begin(), unbox(self).end());
            }
            catch (const std::exception& e) {
                return raise_from_cxx_exception(e);
            }
        };
    }

public:
    /** Add a `__iter__` which produces objects of a further `autoclass` generated type
        that holds onto the iterator-sentinel pair for this type.
    */
    autoclass& iter() {
        unsafe_api();
        require_uninitialized(
            "cannot add iteration methods after class has been created");

        return add_slot(Py_tp_iter, get_iter_func<T>());
    }

private:
    template<typename U>
    hashfunc get_hash_func() {
        return [](PyObject* self) -> Py_hash_t {
            try {
                Py_hash_t hash_value = std::hash<U>{}(unbox(self));
                if (hash_value == -1) {
                    // Python uses -1 to signal failure, but whatever random hash func
                    // we have might actually return that. If this happens, just switch
                    // to -2. This is like how Python uses `id` as hash for int, but
                    // special cases `hash(-1) == -2`.
                    hash_value = -2;
                }
                return hash_value;
            }
            catch (const std::exception& e) {
                raise_from_cxx_exception(e);
                return -1;
            }
        };
    }

public:
    /** Add a `__hash__` which uses `std::hash<T>::operator()`.
     */
    autoclass& hash() {
        require_uninitialized("cannot add a hash method after class has been created");

        return add_slot(Py_tp_hash, get_hash_func<T>());
    }

private:
    template<typename U, typename Args>
    struct invoke_unpack;

    template<typename U, typename... Args>
    struct invoke_unpack<U, std::tuple<Args...>> {
        using type = std::invoke_result_t<U, Args...>;

        static decltype(auto) call(PyObject* self, Args... args) {
            return (autoclass<U>::unbox(self)(args...));
        }
    };

    template<typename U, typename Args>
    ternaryfunc get_call_func() {
        return [](PyObject* self, PyObject* args, PyObject* kwargs) -> PyObject* {
            return detail::automethodwrapper<invoke_unpack<U, Args>::call, 0, PyObject*>(
                self, args, kwargs);
        };
    }

public:
    /** Add a `__call__` method which defers to `T::operator()`.

        @tparam Args The types of the arguments. This selects the particular overload of
                     `operator()` and is used to generate the method signature for
                     `__call__`.
     */
    template<typename... Args>
    autoclass& callable() {
        require_uninitialized("cannot add a call method after class has been created");

        return add_slot(Py_tp_call, get_call_func<T, std::tuple<Args...>>());
    }

private:
    template<typename U>
    reprfunc get_str_func() {
        return [](PyObject* self) -> PyObject* {
            try {
                std::stringstream s;
                s << unbox(self);
                std::size_t size = s.tellp();
                PyObject* out;
                char* buf;
#if PY_MAJOR_VERSION == 2
                out = PyString_FromStringAndSize(nullptr, size);
                if (!out) {
                    return nullptr;
                }
                buf = PyString_AS_STRING(out);
#else
                out = PyUnicode_New(size, PyUnicode_1BYTE_KIND);
                if (!out) {
                    return nullptr;
                }
                buf = PyUnicode_AsUTF8(out);
#endif
                s.read(buf, size);
                return out;
            }
            catch (const std::exception& e) {
                return raise_from_cxx_exception(e);
            }
        };
    }

public:
    /** Add a `__str__` method which uses `operator<<(std::ostrea&, T)`.
     */
    autoclass& str() {
        require_uninitialized("cannot add a str method after class has been created");

        return add_slot(Py_tp_str, get_str_func<T>());
    }

    private:
    template<auto impl>
    reprfunc get_repr_func() {
        return [](PyObject* self) -> PyObject* {
            try {
                auto res = std::invoke(impl, unbox(self));
                std::size_t size = std::size(res);
                py::scoped_ref<> out;
                char* buf;
#if PY_MAJOR_VERSION == 2
                out = py::scoped_ref(PyString_FromStringAndSize(nullptr, size));
                if (!out) {
                    return nullptr;
                }
                buf = PyString_AS_STRING(out);
#else
                out = py::scoped_ref(PyUnicode_New(size, PyUnicode_1BYTE_KIND));
                if (!out) {
                    return nullptr;
                }
                buf = PyUnicode_AsUTF8(out.get());
#endif
                std::copy(std::begin(res), std::end(res), buf);
                return std::move(out).escape();
            }
            catch (const std::exception& e) {
                return raise_from_cxx_exception(e);
            }
        };
    }

public:
    /** Add a `__repr__` method which uses the user provided function. The
        function may either be a member function of `T`, or a free function that
        takes a `T&`. The function must return an iterable of characters to be
        interpreted as utf-8.
     */
    template<auto impl>
    autoclass& repr() {
        require_uninitialized("cannot add a str method after class has been created");

        return add_slot(Py_tp_repr, get_repr_func<impl>());
    }

private:
    /** Helper function to be registered to a weakref that will clear `T` from the type
        cache.
     */
    static void cache_cleanup(PyObject*, PyObject*) {
        detail::autoclass_type_cache.get().erase(typeid(T));
    }

public:
    /** Get a reference to the Python type that represents a boxed `T`.

        This function always returns a non-null pointer, but may throw an exception.

        @note If an exception is thrown, the state of the `autoclass` object is
              unspecified.
     */
    scoped_ref<PyTypeObject> type() {
        if (m_type) {
            return m_type;
        }

        bool have_new = false;
        bool have_traverse = false;
        bool have_clear = false;
        for (PyType_Slot& sl : m_slots) {
            if (sl.slot == Py_tp_new) {
                have_new = true;
            }

            if (sl.slot == Py_tp_traverse) {
                have_traverse = true;
            }

            if (sl.slot == Py_tp_clear) {
                have_clear = true;
            }
        }

        if (have_gc() && !have_traverse &&
            !(m_py_basetype && m_py_basetype->tp_traverse)) {
            throw py::exception(PyExc_ValueError,
                                "if (flags & Py_TPFLAGS_HAVE_GC), a Py_tp_traverse "
                                "slot must be added or be present on the py_basetype");
        }
        if (have_gc()) {
            if (!have_traverse) {
                traverseproc impl =
                    [](PyObject* self, visitproc visit, void* arg) -> int {
                    return self->ob_type->tp_base->tp_traverse(self, visit, arg);
                };
                add_slot(Py_tp_traverse, impl);
            }

            if (!have_clear) {
                inquiry impl = [](PyObject* self) -> int {
                    return self->ob_type->tp_base->tp_clear(self);
                };
                add_slot(Py_tp_clear, impl);
            }
        }

        // cap off our methods
        m_storage->methods.emplace_back(end_method_list);

        // Move our type storage into this persistent storage. `m_storage->methods` has
        // pointers into `m_storage->strings`, so we need to move the list as well to
        // maintain these references.
        auto [it, inserted] =
            detail::autoclass_type_cache.get().try_emplace(typeid(T),
                                                           std::move(m_storage));
        if (!inserted) {
            throw py::exception(PyExc_RuntimeError, "type already constructed");
        }
        std::unique_ptr<detail::autoclass_storage>& storage = it->second;

        // if we need to exit early, evict this cache entry
        py::util::scope_guard release_type_cache(
            [&] { detail::autoclass_type_cache.get().erase(typeid(T)); });

        // Make the `Py_tp_methods` the newly created vector's data, not the original
        // data.
        add_slot(Py_tp_methods, storage->methods.data());
        finalize_slots();
        m_spec.slots = m_slots.data();

        py::scoped_ref type(reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&m_spec)));
        if (!type) {
            throw py::exception{};
        }
        storage->type = type.get();  // borrowed reference held in the cache

        if (!have_new) {
            // explicitly delete the new (don't inherit from Python's `object`)
            type.get()->tp_new = nullptr;
        }
        // don't inherit init
        type.get()->tp_init = nullptr;

        // Create a `PyCFunctionObject` that will call `cache_cleanup`.
        // `PyCFunctionObject` has a pointer to the input `PyMethodDef` which is why
        // we need to store the methoddef on the type cache storage itself.
        storage->callback_method = automethod<cache_cleanup>("cache_cleanup");
        py::scoped_ref callback_func(
            PyCFunction_NewEx(&storage->callback_method, nullptr, nullptr));
        if (!callback_func) {
            throw py::exception{};
        }
        // Create a weakref that calls `callback_func` (A Python function) when `type`
        // dies. This will take a reference to `callback_func`, and after we leave this
        // scope, it will be the sole owner of that function.
        storage->cleanup_wr = py::scoped_ref(
            PyWeakref_NewRef(static_cast<PyObject*>(type), callback_func.get()));
        if (!storage->cleanup_wr) {
            throw py::exception{};
        }

        // We didn't need to exit early, don't evict `T` from the type cache.
        release_type_cache.dismiss();

        m_type = type;
        return type;
    }

    /** Base class for registering a `to_object` handler for this type.

        ### Usage

        ```
        namespace py::dispatch {
        template<>
        struct to_object<C> : public py::autoclass<C>::to_object {};
        }  // namespace py::dispatch
        ```
         */
    class to_object {
        template<typename U>
        static PyObject* f(U&& value) {
            py::scoped_ref<PyTypeObject> cls = lookup_type();
            if (!cls) {
                py::raise(PyExc_RuntimeError) << "autoclass type wasn't initialized yet";
                return nullptr;
            }
            if (cls->tp_flags & Py_TPFLAGS_HAVE_GC) {
                return constructor_new_impl<true>(cls.get(), std::forward<U>(value));
            }
            else {
                return constructor_new_impl<false>(cls.get(), std::forward<U>(value));
            }
        }
    };
};

template<typename T>
int tp_clear(T* ob) {
    return Py_TYPE(ob)->tp_clear(reinterpret_cast<PyObject*>(ob));
}

namespace detail {
/** Initialize the PyBaseExceptionObject component of an autoclass generated
    Python object.
 */
inline void initialize_exception_base(PyBaseExceptionObject* self) {
    zero_non_pyobject_base(self);
    if (!(self->args = PyTuple_New(0))) {
        throw py::exception{};
    }
}
}  // namespace detail

/** A helper for creating autoclasses that are subclasses of a Python exception.
 */
template<typename T>
using exception_autoclass = autoclass<T,
                                      PyBaseExceptionObject,
                                      detail::initialize_exception_base,
                                      tp_clear<PyBaseExceptionObject>>;
}  // namespace py
