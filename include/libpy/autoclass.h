#pragma once

#include <Python.h>
#if PY_MAJOR_VERSION != 2

#include <forward_list>
#include <sstream>
#include <type_traits>
#include <typeindex>
#include <vector>

#include "libpy/automethod.h"
#include "libpy/demangle.h"
#include "libpy/detail/autoclass_cache.h"
#include "libpy/detail/autoclass_object.h"
#include "libpy/meta.h"

namespace py {
template<typename T>
class autoclass {
public:
    using object = detail::autoclass_object<T>;

    template<typename U>
    static T& unbox(const U& ptr) {
        return object::unbox(ptr);
    }

private:
    std::forward_list<std::string> m_strings;
    PyType_Spec m_spec;
    std::vector<PyType_Slot> m_slots;
    scoped_ref<> m_type = nullptr;
    std::vector<PyMethodDef> m_methods;

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
    template<typename R, typename... Args, auto impl>
    struct member_function<R (T::*)(Args...), impl>
        : public pointer_to_member_function_base<impl, R, Args...> {};

    // dispatch for non-const noexcept member function
    template<typename R, typename... Args, auto impl>
    struct member_function<R (T::*)(Args...) noexcept, impl>
        : public pointer_to_member_function_base<impl, R, Args...> {};

    // dispatch for const member function
    template<typename R, typename... Args, auto impl>
    struct member_function<R (T::*)(Args...) const, impl>
        : public pointer_to_member_function_base<impl, R, Args...> {};

    // dispatch for const noexcept member function
    template<typename R, typename... Args, auto impl>
    struct member_function<R (T::*)(Args...) const noexcept, impl>
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

    /** Assert that `m_type` isn't yet initialized. Many operations like adding methods
        will not have any affect after the type is initialized.

        @param msg The error message to forward to the `ValueError` thrown if this
        assertion is violated.
     */
    void require_uninitialized(const char* msg) {
        if (m_type) {
            throw py::exception(PyExc_ValueError, msg);
        }
    }

protected:
    template<typename>
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
    static py::scoped_ref<> lookup_type() {
        auto type_search = detail::autoclass_type_cache.find(typeid(T));
        if (type_search != detail::autoclass_type_cache.end()) {
            return type_search->second;
        }
        return nullptr;
    }

    autoclass(std::string name = util::type_name<T>().get(), int extra_flags = 0)
        : m_strings({std::move(name)}),
          m_spec({m_strings.front().data(),
                  static_cast<int>(sizeof(object)),
                  0,
                  static_cast<unsigned int>(Py_TPFLAGS_DEFAULT | extra_flags),
                  nullptr}) {
        auto type_search = detail::autoclass_type_cache.find(typeid(T));
        if (type_search != detail::autoclass_type_cache.end()) {
            throw std::runtime_error{"type was already created"};
        }

        void (*dealloc)(PyObject*);
        if (have_gc()) {
            dealloc = [](PyObject* self) {
                PyObject_GC_UnTrack(self);
                unbox(self).~T();
                PyObject_GC_Del(self);
            };
        }
        else {
            dealloc = [](PyObject* self) {
                unbox(self).~T();
                PyObject_Del(self);
            };
        }
        add_slot(Py_tp_dealloc, dealloc);
    }

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

        // bind the result of `member_function` to a `traverseproc` to ensure we have
        // a properly typed function before casting to `void*`.
        traverseproc p = member_function<decltype(impl), impl>::f;
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

        // bind the result of `member_function` to an `inquiry` to ensure we have
        // a properly typed function before casting to `void*`.
        inquiry p = member_function<decltype(impl), impl>::f;
        return add_slot(Py_tp_clear, p);
    }

    /** Add a docstring to this class.

        @param doc
        @return *this.
     */
    autoclass& doc(std::string doc) {
        require_uninitialized("cannot add docstring after the class has been created");

        std::string& copied_doc = m_strings.emplace_front(std::move(doc));
        return add_slot(Py_tp_doc, copied_doc.data());
    }

private:
    /** Helper for adapting a function which constructs a `T` into a Python `__new__`
        implementation.
    */
    template<bool have_gc, typename F, auto impl>
    struct free_func_new_impl;

    template<bool have_gc, typename R, typename... Args, auto impl>
    struct free_func_new_impl<have_gc, R(Args...), impl> {
        static PyObject* f(PyTypeObject* cls, Args... args) {
            py::scoped_ref<object> self;
            if (have_gc) {
                self = py::scoped_ref(PyObject_GC_New(object, cls));
            }
            else {
                self = py::scoped_ref(PyObject_New(object, cls));
            }
            if (!self) {
                return nullptr;
            }
            new (&self->cxx_ob) T(impl(std::forward<Args>(args)...));

            if (have_gc) {
                PyObject_GC_Track(self.get());
            }
            return reinterpret_cast<PyObject*>(std::move(self).escape());
        }
    };

    /** Function which will be used to expose one of `T`'s constructors as the `__new__`.
     */
    template<bool have_gc, typename... ConstructorArgs>
    static PyObject* constructor_new_impl(PyTypeObject* cls, ConstructorArgs... args) {
        py::scoped_ref<object> self;
        if (have_gc) {
            self = py::scoped_ref(PyObject_GC_New(object, cls));
        }
        else {
            self = py::scoped_ref(PyObject_New(object, cls));
        }

        if (!self) {
            return nullptr;
        }
        new (&self->cxx_ob) T(std::forward<ConstructorArgs>(args)...);

        if (have_gc) {
            PyObject_GC_Track(self.get());
        }
        return reinterpret_cast<PyObject*>(std::move(self).escape());
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
        require_uninitialized("cannot add size method after the class has been created");
        return add_slot(Py_mp_length, get_length_func<T>());
    }

private:
    /** Template to filter `RHS` down to only the valid types which may appear on the RHS
        of some binary operator.

        @tparam F A template that encodes the operator to check. This should be one of
                  `test_binop<op>::template check`
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
                raise(PyExc_SystemError) << "invalid richcompare op: " << cmp;
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
        @param name The name of the function as it will appear in Python.
        @return *this.
     */
    template<auto impl>
    autoclass& def(std::string name) {
        require_uninitialized("cannot add methods after the class has been created");

        std::string& name_copy = m_strings.emplace_front(std::move(name));
        m_methods.emplace_back(
            automethod<member_function<decltype(impl), impl>::f>(name_copy.data()));
        return *this;
    }

    /** Add a method to this class.

        @tparam impl The implementation of the method to add. If `impl` is a
                pointer to member function, it doesn't need to have the implicit
                `PyObject* self` argument, it will just be called on the boxed value of
                `self`.
        @param name The name of the function as it will appear in Python.
        @param doc The docstring of the function as it will appear in Python.
        @return *this.
     */
    template<auto impl>
    autoclass& def(std::string name, std::string doc) {
        require_uninitialized("cannot add methods after the class has been created");

        std::string& name_copy = m_strings.emplace_front(std::move(name));
        std::string& doc_copy = m_strings.emplace_front(std::move(doc));
        m_methods.emplace_back(
            automethod<member_function<decltype(impl), impl>::f>(name_copy.data(),
                                                                 doc_copy.data()));
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
                auto cls_search = detail::autoclass_type_cache.find(typeid(iter));
                if (cls_search == detail::autoclass_type_cache.end()) {
                    py::raise(PyExc_RuntimeError)
                        << "no iterator type found for " << util::type_name<T>().get();
                    return nullptr;
                }
                auto* cls = reinterpret_cast<PyTypeObject*>(cls_search->second.get());

                Py_INCREF(self);
                py::scoped_ref self_ref(self);
                return autoclass<iter>::template constructor_new_impl<true>(
                    cls, self_ref, unbox(self).begin(), unbox(self).end());
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
            if (kwargs && PyDict_Size(kwargs)) {
                raise(PyExc_TypeError)
                    << Py_TYPE(self)->tp_name << " does not accept keyword arguments";
                return nullptr;
            }
            return detail::automethodwrapper<invoke_unpack<U, Args>::call, 0>(self, args);
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
    reprfunc get_repr_func() {
        return [](PyObject* self) -> PyObject* {
            try {
                std::stringstream s;
                s << unbox(self);
                std::size_t size = s.tellp();
                PyObject* out = PyUnicode_New(size, PyUnicode_1BYTE_KIND);
                if (!out) {
                    return nullptr;
                }
                char* buf = PyUnicode_AsUTF8(out);
                s.read(buf, size);
                return out;
            }
            catch (const std::exception& e) {
                return raise_from_cxx_exception(e);
            }
        };
    }

public:
    /** Add a `__repr__` method which uses `operator<<(std::ostrea&, T)`.
     */
    autoclass& repr() {
        require_uninitialized("cannot add a repr method after class has been created");

        return add_slot(Py_tp_repr, get_repr_func<T>());
    }

    /** Get a reference to the Python type that represents a boxed `T`.
     */
    scoped_ref<> type() {
        if (m_type) {
            return m_type;
        }

        if (have_gc()) {
            bool have_traverse = false;
            for (PyType_Slot& sl : m_slots) {
                if (sl.slot == Py_tp_traverse) {
                    have_traverse = true;
                    break;
                }
            }

            if (!have_traverse) {
                throw py::exception(PyExc_ValueError,
                                    "if (flags & Py_TPFLAGS_HAVE_GC), a Py_tp_traverse "
                                    "slot must be added");
            }
        }

        m_methods.emplace_back(end_method_list);
        auto& storage = detail::autoclass_storage_cache.emplace_front(
            detail::autoclass_storage{std::move(m_methods), std::move(m_strings)});
        add_slot(Py_tp_methods, storage.methods.data());
        finalize_slots();

        m_spec.slots = m_slots.data();
        m_type = scoped_ref(PyType_FromSpec(&m_spec));
        if (!m_type) {
            return nullptr;
        }

        detail::autoclass_type_cache[typeid(T)] = m_type;
        return m_type;
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
            py::scoped_ref<> cls = lookup_type();
            if (!cls) {
                py::raise(PyExc_RuntimeError) << "autoclass type wasn't initialized yet";
                return nullptr;
            }
            PyTypeObject* cls_ob = reinterpret_cast<PyTypeObject*>(cls);
            if (cls_ob->tp_flags & Py_TPFLAGS_HAVE_GC) {
                return constructor_new_impl<true>(cls_ob, std::forward<U>(value));
            }
            else {
                return constructor_new_impl<false>(cls_ob, std::forward<U>(value));
            }
        }
    };
};
}  // namespace py
#endif
