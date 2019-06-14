#pragma once

#include <Python.h>
#if PY_MAJOR_VERSION != 2

#include <sstream>
#include <type_traits>
#include <typeindex>
#include <vector>

#include "libpy/automethod.h"
#include "libpy/demangle.h"
#include "libpy/detail/autoclass_cache.h"

namespace py {
template<typename T>
class autoclass {
public:
    /** The actual type of the Python objects that box a `T`.
     */
    struct object {
        PyObject base;
        T cxx_ob;
    };

    static T& unbox(PyObject* self) {
        return reinterpret_cast<object*>(self)->cxx_ob;
    }

    static T& unbox(const scoped_ref<>& self) {
        return unbox(self.get());
    }

    static T& unbox(const scoped_ref<object>& self) {
        return unbox(reinterpret_cast<PyObject*>(self.get()));
    }

private:
    std::string m_name;
    PyType_Spec m_spec;
    std::vector<PyType_Slot> m_slots;
    scoped_ref<> m_type = nullptr;
    std::vector<PyMethodDef> m_methods;

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

    /** Helper for adapting a function which constructs a `T` into a Python `__new__`
        implementation.
     */
    template<typename F, auto impl>
    struct new_impl;

    template<typename R, typename... Args, auto impl>
    struct new_impl<R(Args...), impl> {
        static PyObject* f(PyTypeObject* cls, Args... args) {
            py::scoped_ref self(PyObject_New(object, cls));
            if (!self) {
                return nullptr;
            }
            new (&self->cxx_ob) T(impl(std::forward<Args>(args)...));
            return reinterpret_cast<PyObject*>(std::move(self).escape());
        }
    };

    /** Function which will be used to expose one of `T`'s constructors as the `__new__`.
     */
    template<typename... ConstructorArgs>
    static PyObject* autonew_impl(PyTypeObject* cls, ConstructorArgs... args) {
        py::scoped_ref self(PyObject_New(object, cls));
        if (!self) {
            return nullptr;
        }
        new (&self->cxx_ob) T(std::forward<ConstructorArgs>(args)...);
        return reinterpret_cast<PyObject*>(std::move(self).escape());
    }

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
    autoclass(const std::string_view& name, int extra_flags = 0u)
        : m_name(name),
          m_spec({nullptr,
                  static_cast<int>(sizeof(object)),
                  0,
                  static_cast<unsigned int>(Py_TPFLAGS_DEFAULT | extra_flags),
                  nullptr}) {
        auto type_search = detail::autoclass_type_cache.find(typeid(T));
        if (type_search != detail::autoclass_type_cache.end()) {
            // the type was already constructed
            m_type = type_search->second;
        }
        else {
            void (*dealloc)(PyObject*) = [](PyObject* self) {
                reinterpret_cast<object*>(self)->cxx_ob.~T();
                PyObject_Del(self);
            };
            add_slot(Py_tp_dealloc, dealloc);
        }
    }

    /** Add a docstring to this class.

        @param doc
        @return *this.
     */
    autoclass& doc(const char* doc) {
        require_uninitialized("cannot add docstring after the class has been created");
        return add_slot(Py_tp_doc, doc);
    }

    /** Add a `__new__` function to this class.

        @tparam impl A function which returns a value which can be used to construct a
                `T`.
        @return *this.
     */
    template<auto impl>
    autoclass& new_() {
        require_uninitialized("cannot add __new__ after the class has been created");
        return add_slot(Py_tp_new, wrap_new<new_impl<decltype(impl), impl>::f>);
    }

private:
    template<typename U,
             typename =
                 std::void_t<decltype(static_cast<Py_ssize_t>(std::declval<U>().size()))>>
    lenfunc get_length_func(int) {
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

    template<typename>
    lenfunc get_length_func(long) {
        return nullptr;
    }

public:
    /** Add a `__len__` method from `T::size()`
     */
    autoclass& size() {
        require_uninitialized("cannot add size method after the class has been created");
        return add_slot(Py_mp_length, get_length_func<T>(0));
    }

    /** Add a `__new__` function to this class by adapting one of the constructors of `T`.

        @tparam ConstructorArgs The C++ signature of the constructor to use.
        @return *this.
     */
    template<typename... ConstructorArgs>
    autoclass& new_() {
        require_uninitialized("cannot add __new__ after the class has been created");
        return add_slot(Py_tp_new, wrap_new<autonew_impl<ConstructorArgs...>>);
    }

private:
    template<template<typename, typename> typename F, typename RHS>
    struct valid_rhs_types;

    template<template<typename, typename> typename F, typename Head, typename... Tail>
    struct valid_rhs_types<F, std::tuple<Head, Tail...>> {
    private:
        using rest = std::tuple<Tail...>;

    public:
        using type = std::conditional_t<
            (std::is_same_v<Head, T> || py::has_from_object<Head>) &&F<T, Head>::value,
            meta::type_cat<std::tuple<Head>, typename valid_rhs_types<F, rest>::type>,
            typename valid_rhs_types<F, rest>::type>;
    };

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

    template<typename RHS, typename F, typename LHS>
    static PyObject* search_binop_implementations(F&& f, LHS& lhs, PyObject* rhs) {
        return search_binop_implementations_helper<F, LHS, RHS>::f(std::forward<F>(f),
                                                                   lhs,
                                                                   rhs);
    }

#define DEFINE_GET_BINOP(op, name)                                                       \
    template<typename L, typename R>                                                     \
    struct test_binop_##name {                                                           \
    private:                                                                             \
        template<typename LHS, typename RHS>                                             \
        static decltype(std::declval<LHS>() op std::declval<RHS>(), std::true_type{})    \
        valid(int);                                                                      \
                                                                                         \
        template<typename, typename>                                                     \
        static std::false_type valid(long);                                              \
                                                                                         \
    public:                                                                              \
        static constexpr bool value =                                                    \
            std::is_same_v<decltype(valid<L, R>(0)), std::true_type>;                    \
    };                                                                                   \
                                                                                         \
    template<typename U,                                                                 \
             typename RHS,                                                               \
             typename = std::enable_if_t<!std::is_same_v<                                \
                 std::tuple<>,                                                           \
                 typename valid_rhs_types<test_binop_##name, RHS>::type>>>               \
    static constexpr auto get_##name##_func_impl(int) {                                  \
        using valid_rhs = typename valid_rhs_types<test_binop_##name, RHS>::type;        \
        return [](PyObject* self, PyObject* other) -> PyObject* {                        \
            try {                                                                        \
                if (meta::element_of<T, valid_rhs> && Py_TYPE(self) == Py_TYPE(other)) { \
                    return py::to_object(unbox(self) op unbox(other)).escape();          \
                }                                                                        \
                else {                                                                   \
                    using rhs_without_T = meta::set_diff<valid_rhs, std::tuple<T>>;      \
                    return search_binop_implementations<rhs_without_T>(                  \
                        [](auto& a, const auto& b) { return a op b; },                   \
                        unbox(self),                                                     \
                        other);                                                          \
                }                                                                        \
            }                                                                            \
            catch (const std::exception& e) {                                            \
                return raise_from_cxx_exception(e);                                      \
            }                                                                            \
        };                                                                               \
    }                                                                                    \
                                                                                         \
    template<typename, typename>                                                         \
    static constexpr binaryfunc get_##name##_func_impl(long) {                           \
        return nullptr;                                                                  \
    }                                                                                    \
                                                                                         \
    template<typename LHS, typename RHS>                                                 \
    static constexpr binaryfunc name##_arith_func = get_##name##_func_impl<LHS, RHS>(0); \
                                                                                         \
    template<typename LHS, typename RHS>                                                 \
    static constexpr auto get_##name##_cmp_func() {                                      \
        auto out = get_##name##_func_impl<LHS, RHS>(0);                                  \
        if constexpr (std::is_same_v<decltype(out), binaryfunc>) {                       \
            return [](PyObject*, PyObject*) { Py_RETURN_NOTIMPLEMENTED; };               \
        }                                                                                \
        else {                                                                           \
            return out;                                                                  \
        }                                                                                \
    }                                                                                    \
                                                                                         \
    template<typename LHS, typename RHS>                                                 \
    static constexpr auto name##_cmp_func = get_##name##_cmp_func<LHS, RHS>();

    DEFINE_GET_BINOP(+, add)
    DEFINE_GET_BINOP(-, sub)
    DEFINE_GET_BINOP(*, mul)
    DEFINE_GET_BINOP(%, rem)
    DEFINE_GET_BINOP(/, floor_divide)
    DEFINE_GET_BINOP(<<, lshift)
    DEFINE_GET_BINOP(>>, rshift)
    DEFINE_GET_BINOP(&, and)
    DEFINE_GET_BINOP(^, xor)
    DEFINE_GET_BINOP(|, or)
    DEFINE_GET_BINOP(>, gt)
    DEFINE_GET_BINOP(>=, ge)
    DEFINE_GET_BINOP(==, eq)
    DEFINE_GET_BINOP(<=, le)
    DEFINE_GET_BINOP(<, lt)
    DEFINE_GET_BINOP(!=, ne)

#undef DEFINE_GET_BINOP

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

        add_slot(Py_nb_add, add_arith_func<T, RHS>);
        add_slot(Py_nb_subtract, sub_arith_func<T, RHS>);
        add_slot(Py_nb_multiply, mul_arith_func<T, RHS>);
        add_slot(Py_nb_remainder, rem_arith_func<T, RHS>);
        add_slot(Py_nb_floor_divide, floor_divide_arith_func<T, RHS>);
        add_slot(Py_nb_lshift, lshift_arith_func<T, RHS>);
        add_slot(Py_nb_rshift, rshift_arith_func<T, RHS>);
        add_slot(Py_nb_and, and_arith_func<T, RHS>);
        add_slot(Py_nb_xor, xor_arith_func<T, RHS>);
        add_slot(Py_nb_or, or_arith_func<T, RHS>);
        return *this;
    }

private:
    template<typename RHS>
    static PyObject* richcompare_impl(PyObject* self, PyObject* other, int cmp) {
        try {
            switch (cmp) {
            case Py_LT:
                return lt_cmp_func<T, RHS>(self, other);
            case Py_LE:
                return le_cmp_func<T, RHS>(self, other);
            case Py_EQ:
                return eq_cmp_func<T, RHS>(self, other);
            case Py_GE:
                return ge_cmp_func<T, RHS>(self, other);
            case Py_GT:
                return gt_cmp_func<T, RHS>(self, other);
            case Py_NE:
                return ne_cmp_func<T, RHS>(self, other);
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
#define DEFINE_GET_UNOP(op, name)                                                        \
    template<typename U, typename = std::void_t<decltype(op std::declval<U>())>>         \
    unaryfunc get_##name##_func(int) {                                                   \
        return [](PyObject* self) -> PyObject* {                                         \
            try {                                                                        \
                return py::to_object(op unbox(self)).escape();                           \
            }                                                                            \
            catch (const std::exception& e) {                                            \
                return raise_from_cxx_exception(e);                                      \
            }                                                                            \
        };                                                                               \
    }                                                                                    \
                                                                                         \
    template<typename>                                                                   \
    static constexpr unaryfunc get_##name##_func(long) {                                 \
        return nullptr;                                                                  \
    }

    DEFINE_GET_UNOP(-, neg)
    DEFINE_GET_UNOP(+, pos)
    DEFINE_GET_UNOP(~, inv)

#undef DEFINE_GET_UNNOP

public:
    /** Add unary operator methods `__neg__` and `__pos__`, and `__inv__` from the C++
        `operator-`, `operator+`, and `operator~` respectively.  interface.

     */
    autoclass& unary() {
        require_uninitialized(
            "cannot add unary operator methods after the class has been created");

        add_slot(Py_nb_negative, get_neg_func<T>(0));
        add_slot(Py_nb_positive, get_pos_func<T>(0));
        add_slot(Py_nb_invert, get_inv_func<T>(0));
        return *this;
    }

private:
#define DEFINE_CONVERSION_OP(type, name)                                                 \
    template<typename U,                                                                 \
             typename = std::void_t<decltype(static_cast<type>(std::declval<U>()))>>     \
    unaryfunc get_convert_##name##_func(int) {                                           \
        return [](PyObject* self) -> PyObject* {                                         \
            try {                                                                        \
                return py::to_object(static_cast<type>(unbox(self))).escape();           \
            }                                                                            \
            catch (const std::exception& e) {                                            \
                return raise_from_cxx_exception(e);                                      \
            }                                                                            \
        };                                                                               \
    }                                                                                    \
                                                                                         \
    template<typename>                                                                   \
    unaryfunc get_convert_##name##_func(long) {                                          \
        return nullptr;                                                                  \
    }

    DEFINE_CONVERSION_OP(std::int64_t, int)
    DEFINE_CONVERSION_OP(double, float)

#undef DEFINE_CONVERSION_OP

    template<typename U,
             typename = std::void_t<decltype(static_cast<bool>(std::declval<U>()))>>
    inquiry get_convert_bool_func(int) {
        return [](PyObject* self) -> int {
            try {
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
        add_slot(Py_nb_int, get_convert_int_func<T>(0));
        add_slot(Py_nb_float, get_convert_float_func<T>(0));
        add_slot(Py_nb_bool, get_convert_bool_func<T>(0));
        return *this;
    }

private:
    template<typename U,
             typename KeyType,
             typename = std::void_t<decltype(std::declval<U>()[std::declval<KeyType>()])>>
    binaryfunc get_getitem_func(int) {
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

    template<typename U, typename KeyType>
    binaryfunc get_getitem_func(long) {
        return nullptr;
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
        add_slot(Py_mp_subscript, get_getitem_func<T, KeyType>(0));
        add_slot(Py_mp_ass_subscript, get_setitem_func<T, KeyType, ValueType>(0));
        return size();
    }

    /** Add a method to this class.

        @tparam impl The implementation of the method to add. If `impl` is a
       pointer to member function, it doesn't need to have the implicit
       `PyObject* self` argument, it will just be called on the boxed value of
       `self`.
        @param name The name of the function as it will appear in Python.
        @param doc The docstring for this function.
        @return *this.
     */
    template<auto impl>
    autoclass& def(const char* name, const char* doc = nullptr) {
        require_uninitialized("cannot add methods after the class has been created");
        m_methods.emplace_back(
            automethod<member_function<decltype(impl), impl>::f>(name, doc));
        return *this;
    }

private:
    template<typename U,
             typename = std::void_t<decltype(std::declval<U>().begin(),
                                             std::declval<U>().end())>>
    getiterfunc get_iter_func(int) {
        // new-style ranges allow for begin and end to produce different types
        using begin_type = std::remove_reference_t<decltype(std::declval<T>().begin())>;
        using end_type = std::remove_reference_t<decltype(std::declval<T>().end())>;

        struct iter {
            begin_type it;
            end_type end;
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

        std::string iter_name = util::type_name<T>().get();
        iter_name += "::iterator";
        // create the iterator class and put it in the cache
        if (!autoclass<iter>(iter_name.data())
                 .add_slot(Py_tp_iternext, static_cast<iternextfunc>(iternext))
                 .add_slot(Py_tp_iter, &PyObject_SelfIter)
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
                using object = typename autoclass<iter>::object;
                py::scoped_ref it(PyObject_New(object, cls));
                if (!it) {
                    return nullptr;
                }
                iter& unboxed = autoclass<iter>::unbox(it);
                new (&unboxed.it) begin_type(unbox(self).begin());
                new (&unboxed.end) end_type(unbox(self).end());
                return reinterpret_cast<PyObject*>(std::move(it).escape());
            }
            catch (const std::exception& e) {
                return raise_from_cxx_exception(e);
            }
        };
    }

    template<typename>
    getiterfunc get_iter_func(long) {
        return nullptr;
    }

public:
    /** Add a `__iter__` which produces objects of a further `autoclass` generated type
        that holds onto the iterator-sentinel pair for this type.
    */
    autoclass& range() {
        require_uninitialized(
            "cannot add iteration methods after class has been created");

        return add_slot(Py_tp_iter, get_iter_func<T>(0));
    }

private:
    template<typename U, typename = std::hash<U>>
    hashfunc get_hash_func(int) {
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

    template<typename>
    hashfunc get_hash_func(long) {
        return nullptr;
    }

public:
    /** Add a `__hash__` which uses `std::hash<T>::operator()`.
     */
    autoclass& hash() {
        require_uninitialized("cannot add a hash method after class has been created");

        return add_slot(Py_tp_hash, get_hash_func<T>(0));
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

    template<typename U, typename Args, typename = typename invoke_unpack<U, Args>::type>
    ternaryfunc get_call_func(int) {
        return [](PyObject* self, PyObject* args, PyObject* kwargs) -> PyObject* {
            if (kwargs && PyDict_Size(kwargs)) {
                raise(PyExc_TypeError)
                    << Py_TYPE(self)->tp_name << " does not accept keyword arguments";
                return nullptr;
            }
            return detail::automethodwrapper<invoke_unpack<U, Args>::call, 0>(self, args);
        };
    }

    template<typename, typename>
    ternaryfunc get_call_func(long) {
        return nullptr;
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

        return add_slot(Py_tp_call, get_call_func<T, std::tuple<Args...>>(0));
    }

private:
    template<typename U,
             typename = std::void_t<decltype(std::declval<std::ostream&>()
                                             << std::declval<U>())>>
    reprfunc get_repr_func(int) {
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

    template<typename>
    reprfunc get_repr_func(long) {
        return nullptr;
    }

public:
    /** Add a `__repr__` method which uses `operator<<(std::ostrea&, T)`.
     */
    autoclass& repr() {
        require_uninitialized("cannot add a repr method after class has been created");

        return add_slot(Py_tp_repr, get_repr_func<T>(0));
    }

    /** Get a reference to the Python type that represents a boxed `T`.
     */
    scoped_ref<> type() {
        if (m_type) {
            return m_type;
        }

        m_methods.emplace_back(end_method_list);
        auto& storage = detail::autoclass_storage_cache.emplace_back(
            detail::autoclass_storage{std::move(m_methods), std::move(m_name)});
        add_slot(Py_tp_methods, storage.methods.data());
        finalize_slots();

        m_spec.name = storage.name.data();
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
        using C = ...;
        py::scoped_ref cls = py::autoclass<C>.type();

        namespace py::dispatch {
        template<>
        struct to_object<C> : public py::autoclass<C>::to_object<cls.get()> {};
        }  // namespace py::dispatch
        ```
     */
    template<PyObject* cls>
    class to_object {
        static PyObject* f(const T& value) {
            return autonew_impl(reinterpret_cast<PyTypeObject*>(cls), value);
        }

        static PyObject* f(T&& value) {
            return autonew_impl(reinterpret_cast<PyTypeObject*>(cls), std::move(value));
        }
    };
};
}  // namespace py
#endif
