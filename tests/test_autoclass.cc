#include <Python.h>

#include <map>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

#include "libpy/autoclass.h"
#include "libpy/call_function.h"
#include "libpy/from_object.h"
#include "test_utils.h"

namespace test_autoclass {
using namespace std::literals;

class autoclass : public with_python_interpreter {
    void TearDown() override {
        // Types basically always participate in a cycle because the method descriptors.
        // Run the collector until there is no more garbage.
        while (PyGC_Collect())
            ;
        EXPECT_EQ(py::detail::autoclass_type_cache.size(), 0ul);

        with_python_interpreter::TearDown();
    }
};

TEST_F(autoclass, smoke) {
    class C {
    private:
        int m_a;
        float m_b;

    public:
        C(int a, float b) : m_a(a), m_b(b) {}

        int a() const {
            return m_a;
        }

        float b() const {
            return m_b;
        }

        float sum() const {
            return m_a + m_b;
        }

        float sum_plus(float arg) const {
            return sum() + arg;
        }
    };

    py::scoped_ref cls = py::autoclass<C>()
                             .new_<int, float>()
                             .def<&C::a>("a")
                             .def<&C::b>("b")
                             .def<&C::sum>("sum")
                             .def<&C::sum_plus>("sum_plus")
                             .type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls), 1, 2.5);
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());
    C& unboxed = py::autoclass<C>::unbox(inst);

    {
        auto res_ob = py::call_method(inst, "a");
        ASSERT_TRUE(res_ob);
        int res = py::from_object<int>(res_ob);
        EXPECT_EQ(res, unboxed.a());
    }

    {
        auto res_ob = py::call_method(inst, "b");
        ASSERT_TRUE(res_ob);
        float res = py::from_object<float>(res_ob);
        EXPECT_EQ(res, unboxed.b());
    }

    {
        auto res_ob = py::call_method(inst, "sum");
        ASSERT_TRUE(res_ob);
        float res = py::from_object<float>(res_ob);
        EXPECT_EQ(res, unboxed.sum());
    }

    {
        auto res_ob = py::call_method(inst, "sum_plus", 1.5);
        ASSERT_TRUE(res_ob);
        float res = py::from_object<float>(res_ob);
        EXPECT_EQ(res, unboxed.sum_plus(1.5));
    }
}

TEST_F(autoclass, def_inherited_method) {
    struct C {
        int f(int a) {
            return a + 1;
        }
    };

    struct D : public C {};

    py::scoped_ref cls = py::autoclass<D>().new_<>().def<&D::f>("f").type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls));
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    py::scoped_ref<> res = py::call_method(inst, "f", 1);
    ASSERT_TRUE(res);

    EXPECT_EQ(py::from_object<int>(res), 2);
}

TEST_F(autoclass, construct) {
    struct C {
        int a;
        float b;

        C(int a, float b) : a(a), b(b) {}
    };

    // don't use `new_` to expose `__new__` to Python, `autoclass::construct` doesn't need
    // that
    py::scoped_ref cls = py::autoclass<C>().type();
    ASSERT_TRUE(cls);

    py::scoped_ref<> inst = py::autoclass<C>::construct(1, 2.5);
    ASSERT_TRUE(inst);

    EXPECT_EQ(Py_TYPE(inst.get()), cls.get());
    C& unboxed = py::autoclass<C>::unbox(inst);

    EXPECT_EQ(unboxed.a, 1);
    EXPECT_EQ(unboxed.b, 2.5);
}

TEST_F(autoclass, name) {
    {
        class C {};

        py::scoped_ref cls = py::autoclass<C>().type();
        ASSERT_TRUE(cls);

        const char* const tp_name = reinterpret_cast<PyTypeObject*>(cls.get())->tp_name;
        EXPECT_STREQ(tp_name, py::util::type_name<C>().get());
    }

    {
        class C {};

        py::scoped_ref cls = py::autoclass<C>("custom_name").type();
        ASSERT_TRUE(cls);

        const char* const tp_name = reinterpret_cast<PyTypeObject*>(cls.get())->tp_name;
        EXPECT_STREQ(tp_name, "custom_name");
    }
}

TEST_F(autoclass, doc) {
    {
        class C {};

        py::scoped_ref cls = py::autoclass<C>().type();
        ASSERT_TRUE(cls);

        const char* const tp_doc = reinterpret_cast<PyTypeObject*>(cls.get())->tp_doc;
        EXPECT_EQ(tp_doc, nullptr);
    }

    {
        class C {};

        py::scoped_ref cls = py::autoclass<C>().doc("custom doc").type();
        ASSERT_TRUE(cls);

        const char* const tp_doc = reinterpret_cast<PyTypeObject*>(cls.get())->tp_doc;
        EXPECT_STREQ(tp_doc, "custom doc");
    }
}

#define TEST_AUTOCLASS_BINARY_OPERATOR(op, name, activate, pyfunc)                       \
    TEST_F(autoclass, operator_##name) {                                                 \
        struct C {                                                                       \
            int value;                                                                   \
                                                                                         \
            C(int value) : value(value) {}                                               \
                                                                                         \
            int operator op(const C& other) const {                                      \
                return value op other.value;                                             \
            }                                                                            \
                                                                                         \
            int operator op(int other) const {                                           \
                if (other == -1) {                                                       \
                    throw std::invalid_argument{"C++ message"};                          \
                }                                                                        \
                return value op other;                                                   \
            }                                                                            \
        };                                                                               \
        py::scoped_ref cls = py::autoclass<C>().new_<int>().activate<C, int>().type();   \
        ASSERT_TRUE(cls);                                                                \
                                                                                         \
        py::scoped_ref lhs = py::call_function(static_cast<PyObject*>(cls), 10);         \
        ASSERT_TRUE(lhs);                                                                \
        ASSERT_EQ(Py_TYPE(lhs.get()), cls.get());                                        \
        C& unboxed_lhs = py::autoclass<C>::unbox(lhs);                                   \
                                                                                         \
        {                                                                                \
            py::scoped_ref rhs = py::call_function(static_cast<PyObject*>(cls), 7);      \
            ASSERT_TRUE(rhs);                                                            \
            ASSERT_EQ(Py_TYPE(rhs.get()), cls.get());                                    \
            C& unboxed_rhs = py::autoclass<C>::unbox(rhs);                               \
            py::scoped_ref boxed_result(pyfunc(lhs.get(), rhs.get()));                   \
            ASSERT_TRUE(boxed_result);                                                   \
                                                                                         \
            EXPECT_EQ(py::from_object<int>(boxed_result), unboxed_lhs op unboxed_rhs);   \
        }                                                                                \
                                                                                         \
        {                                                                                \
            int unboxed_rhs = 7;                                                         \
            py::scoped_ref boxed_rhs = py::to_object(unboxed_rhs);                       \
            ASSERT_TRUE(boxed_rhs);                                                      \
            py::scoped_ref boxed_result(pyfunc(lhs.get(), boxed_rhs.get()));             \
            ASSERT_TRUE(boxed_result);                                                   \
                                                                                         \
            EXPECT_EQ(py::from_object<int>(boxed_result), unboxed_lhs op unboxed_rhs);   \
        }                                                                                \
                                                                                         \
        {                                                                                \
            int unboxed_rhs = -1;                                                        \
            py::scoped_ref boxed_rhs = py::to_object(unboxed_rhs);                       \
            ASSERT_TRUE(boxed_rhs);                                                      \
            py::scoped_ref boxed_result(pyfunc(lhs.get(), boxed_rhs.get()));             \
            EXPECT_FALSE(boxed_result);                                                  \
            expect_pyerr_type_and_message(PyExc_RuntimeError,                            \
                                          "a C++ exception was raised: C++ message");    \
            PyErr_Clear();                                                               \
        }                                                                                \
    }

// arithmetic
TEST_AUTOCLASS_BINARY_OPERATOR(+, add, arithmetic, PyNumber_Add)
TEST_AUTOCLASS_BINARY_OPERATOR(-, sub, arithmetic, PyNumber_Subtract)
TEST_AUTOCLASS_BINARY_OPERATOR(*, mul, arithmetic, PyNumber_Multiply)
TEST_AUTOCLASS_BINARY_OPERATOR(/, div, arithmetic, PyNumber_FloorDivide)
TEST_AUTOCLASS_BINARY_OPERATOR(<<, lsjift, arithmetic, PyNumber_Lshift)
TEST_AUTOCLASS_BINARY_OPERATOR(>>, rshift, arithmetic, PyNumber_Rshift)
TEST_AUTOCLASS_BINARY_OPERATOR(&, and, arithmetic, PyNumber_And)
TEST_AUTOCLASS_BINARY_OPERATOR(^, xor, arithmetic, PyNumber_Xor)
TEST_AUTOCLASS_BINARY_OPERATOR(|, or, arithmetic, PyNumber_Or)

// comparisons
TEST_AUTOCLASS_BINARY_OPERATOR(>, gt, comparisons, [](PyObject* lhs, PyObject* rhs) {
    return PyObject_RichCompare(lhs, rhs, Py_GT);
})
TEST_AUTOCLASS_BINARY_OPERATOR(>=, ge, comparisons, [](PyObject* lhs, PyObject* rhs) {
    return PyObject_RichCompare(lhs, rhs, Py_GE);
})
TEST_AUTOCLASS_BINARY_OPERATOR(==, eq, comparisons, [](PyObject* lhs, PyObject* rhs) {
    return PyObject_RichCompare(lhs, rhs, Py_EQ);
})
TEST_AUTOCLASS_BINARY_OPERATOR(<=, le, comparisons, [](PyObject* lhs, PyObject* rhs) {
    return PyObject_RichCompare(lhs, rhs, Py_LE);
})
TEST_AUTOCLASS_BINARY_OPERATOR(<, lt, comparisons, [](PyObject* lhs, PyObject* rhs) {
    return PyObject_RichCompare(lhs, rhs, Py_LT);
})
TEST_AUTOCLASS_BINARY_OPERATOR(!=, ne, comparisons, [](PyObject* lhs, PyObject* rhs) {
    return PyObject_RichCompare(lhs, rhs, Py_NE);
})

#undef TEST_AUTOCLASS_BINARY_OPERATOR

#define TEST_AUTOCLASS_UNARY_OPERATOR(op, name, pyfunc)                                  \
    TEST_F(autoclass, operator_##name) {                                                 \
        struct C {                                                                       \
            int value;                                                                   \
                                                                                         \
            C(int value) : value(value) {}                                               \
                                                                                         \
            int operator op() const {                                                    \
                if (value == -1) {                                                       \
                    throw std::invalid_argument{"C++ message"};                          \
                }                                                                        \
                return op value;                                                         \
            }                                                                            \
        };                                                                               \
        py::scoped_ref cls = py::autoclass<C>().new_<int>().unary().type();              \
        ASSERT_TRUE(cls);                                                                \
                                                                                         \
        {                                                                                \
            py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls), 10);    \
            ASSERT_TRUE(inst);                                                           \
            ASSERT_EQ(Py_TYPE(inst.get()), cls.get());                                   \
                                                                                         \
            py::scoped_ref result(pyfunc(inst.get()));                                   \
            ASSERT_TRUE(result);                                                         \
            int unboxed_result = py::from_object<int>(result.get());                     \
                                                                                         \
            EXPECT_EQ(unboxed_result, op 10);                                            \
        }                                                                                \
                                                                                         \
        {                                                                                \
            py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls), -1);    \
            ASSERT_TRUE(inst);                                                           \
            ASSERT_EQ(Py_TYPE(inst.get()), cls.get());                                   \
                                                                                         \
            py::scoped_ref result(pyfunc(inst.get()));                                   \
            EXPECT_FALSE(result);                                                        \
            expect_pyerr_type_and_message(PyExc_RuntimeError,                            \
                                          "a C++ exception was raised: C++ message");    \
            PyErr_Clear();                                                               \
        }                                                                                \
    }

TEST_AUTOCLASS_UNARY_OPERATOR(-, neg, PyNumber_Negative)
TEST_AUTOCLASS_UNARY_OPERATOR(+, pos, PyNumber_Positive)
TEST_AUTOCLASS_UNARY_OPERATOR(~, inv, PyNumber_Invert)

#undef TEST_AUTOCLASS_UNARY_OPERATOR

template<typename T, bool should_throw>
void test_type_conversion(const char* method_name) {
    struct C {
        int value;

        C(int value) : value(value) {}

        explicit operator T() const {
            if (should_throw) {
                throw std::bad_cast{};
            }
            return static_cast<T>(value);
        }
    };

    py::scoped_ref cls = py::autoclass<C>().template new_<int>().conversions().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls), 10);
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());
    C& unboxed = py::autoclass<C>::unbox(inst);

    if (should_throw) {
        py::scoped_ref boxed_result(py::call_method(inst, method_name));
        EXPECT_FALSE(boxed_result);
        expect_pyerr_type_and_message(PyExc_RuntimeError,
                                      "a C++ exception was raised: std::bad_cast");
        PyErr_Clear();
    }
    else {
        py::scoped_ref boxed_result(py::call_method(inst, method_name));
        ASSERT_TRUE(boxed_result);
        EXPECT_EQ(py::from_object<T>(boxed_result), static_cast<T>(unboxed));
    }
}

TEST_F(autoclass, int_conversion) {
    test_type_conversion<std::int64_t, false>("__int__");
    test_type_conversion<std::int64_t, true>("__int__");
}

TEST_F(autoclass, float_conversion) {
    test_type_conversion<double, false>("__float__");
    test_type_conversion<double, true>("__float__");
}

namespace {
#if PY_MAJOR_VERSION == 2
const char* const bool_method_name = "__nonzero__";
#else
const char* const bool_method_name = "__bool__";
#endif
}

TEST_F(autoclass, bool_conversion) {
    test_type_conversion<bool, false>(bool_method_name);
    test_type_conversion<bool, true>(bool_method_name);
}

TEST_F(autoclass, from_object) {
    struct C {
        int data;

        C(int data) : data(data) {}
    };

    py::scoped_ref cls = py::autoclass<C>().new_<int>().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls), 5);
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    {
        const C& unboxed_ref = py::from_object<const C&>(inst);
        EXPECT_EQ(unboxed_ref.data, 5);
    }

    {
        C& unboxed_ref = py::from_object<C&>(inst);
        EXPECT_EQ(unboxed_ref.data, 5);
        unboxed_ref.data = 6;
    }

    {
        const C& unboxed_ref = py::from_object<const C&>(inst);
        EXPECT_EQ(unboxed_ref.data, 6);
    }
}

TEST_F(autoclass, callable) {
    struct C {
        double value;

        C(double value) : value(value) {}

        double operator()(int a, double b) const {
            return value + a + b;
        }
    };

    py::scoped_ref cls = py::autoclass<C>().new_<double>().callable<int, double>().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls), 1.0);
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    py::scoped_ref result = py::call_function(inst, 2, 3.5);
    ASSERT_TRUE(result);

    double unboxed_result = py::from_object<double>(result);
    EXPECT_EQ(unboxed_result, C{1.0}(2, 3.5));
}

TEST_F(autoclass, callable_throws) {
    struct C {
        void operator()() const {
            throw std::runtime_error{"lmao"};
        }
    };

    py::scoped_ref cls = py::autoclass<C>().new_<>().callable<>().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls));
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    py::scoped_ref result = py::call_function(inst);
    ASSERT_FALSE(result);
    expect_pyerr_type_and_message(PyExc_RuntimeError, "a C++ exception was raised: lmao");
    PyErr_Clear();
}

TEST_F(autoclass, hash) {
    py::scoped_ref cls = py::autoclass<std::string>().new_<std::string>().hash().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls), "ayy lmao");
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    Py_hash_t result = PyObject_Hash(inst.get());
    ASSERT_FALSE(PyErr_Occurred());

    Py_hash_t expected = std::hash<std::string>{}("ayy lmao");
    EXPECT_EQ(result, expected);
}

// used in `hash_throws` test
struct broken_hash_type {};
}  // namespace test_autoclass

namespace std {
template<>
struct hash<test_autoclass::broken_hash_type> {
    [[noreturn]] std::size_t operator()(test_autoclass::broken_hash_type) const {
        throw std::runtime_error{"ayy"};
    }
};
}  // namespace std

namespace test_autoclass {
TEST_F(autoclass, hash_throws) {
    py::scoped_ref cls = py::autoclass<broken_hash_type>().new_<>().hash().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls));
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    Py_hash_t result = PyObject_Hash(inst.get());
    EXPECT_EQ(result, -1);
    expect_pyerr_type_and_message(PyExc_RuntimeError, "a C++ exception was raised: ayy");
    PyErr_Clear();
}

// used in `hash_returns_negative_one` test
struct negative_one_hash {};
}  // namespace test_autoclass

namespace std {
template<>
struct hash<test_autoclass::negative_one_hash> {
    std::size_t operator()(test_autoclass::negative_one_hash) const {
        return -1;
    }
};
}  // namespace std

namespace test_autoclass {
// Python uses -1 as an error sentinel in `tp_hash`, so we need to correct a non-failure
// -1 from our type's hash function to a valid value.
TEST_F(autoclass, hash_returns_negative_one) {
    py::scoped_ref cls = py::autoclass<negative_one_hash>().new_<>().hash().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls));
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    Py_hash_t result = PyObject_Hash(inst.get());
    EXPECT_EQ(result, -2);
}

TEST_F(autoclass, repr) {
    py::scoped_ref cls = py::autoclass<std::string>().new_<std::string>().repr().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls), "ayy lmao");
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    py::scoped_ref repr_ob(PyObject_Repr(inst.get()));
    ASSERT_TRUE(repr_ob);

    EXPECT_EQ(py::util::pystring_to_string_view(repr_ob), "ayy lmao"sv);
}

#if LIBPY_AUTOCLASS_UNSAFE_API
TEST_F(autoclass, len) {
    auto check = [](const py::scoped_ref<PyTypeObject>& cls) {
        ASSERT_TRUE(cls);
        py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls));
        ASSERT_TRUE(inst);
        ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

        Py_ssize_t size = PyObject_Length(inst.get());
        EXPECT_EQ(size, 10);
    };

    {
        struct C {
            std::size_t size() const {
                return 10;
            }
        };

        check(py::autoclass<C>().new_<>().len().type());
    }

    {
        struct C {
            std::size_t size() const {
                return 10;
            }

            int operator[](std::size_t) const {
                return 0;
            }
        };

        // mapping also gives size, if present
        check(py::autoclass<C>().new_<>().mapping<std::size_t>().type());
    }
}

TEST_F(autoclass, len_throws) {
    struct C {
        [[noreturn]] std::size_t size() const {
            throw std::runtime_error{"idk man"};
        }
    };

    py::scoped_ref cls = py::autoclass<C>().new_<>().len().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls));
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    Py_ssize_t size = PyObject_Length(inst.get());
    EXPECT_EQ(size, -1);
    expect_pyerr_type_and_message(PyExc_RuntimeError,
                                  "a C++ exception was raised: idk man");
    PyErr_Clear();
}

TEST_F(autoclass, mapping_getitem) {
    auto check = [](auto type) {
        using M = decltype(type);
        using key_type = typename M::key_type;
        using value_type = typename M::mapped_type;

        py::scoped_ref cls =
            py::autoclass<M>("M").template new_<>().template mapping<key_type>().type();
        ASSERT_TRUE(cls);

        py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls));
        ASSERT_TRUE(inst);
        ASSERT_EQ(Py_TYPE(inst.get()), cls.get());
        M& unboxed = py::autoclass<M>::unbox(inst);

        key_type key_1 = 1;
        value_type value_1 = 1;
        unboxed[key_1] = value_1;

        key_type key_2 = 2;
        value_type value_2 = 2;
        unboxed[key_2] = value_2;

        py::scoped_ref boxed_key_1 = py::to_object(key_1);
        ASSERT_TRUE(boxed_key_1);
        py::scoped_ref boxed_value_1(PyObject_GetItem(inst.get(), boxed_key_1.get()));
        ASSERT_TRUE(boxed_value_1);
        value_type unboxed_value_1 = py::from_object<value_type>(boxed_value_1);
        EXPECT_EQ(unboxed_value_1, value_1);

        py::scoped_ref boxed_key_2 = py::to_object(key_2);
        ASSERT_TRUE(boxed_key_2);
        py::scoped_ref boxed_value_2(PyObject_GetItem(inst.get(), boxed_key_2.get()));
        ASSERT_TRUE(boxed_value_2);
        value_type unboxed_value_2 = py::from_object<value_type>(boxed_value_2);
        EXPECT_EQ(unboxed_value_2, value_2);
    };

    check(std::map<int, int>{});
    check(std::map<char, int>{});
    check(std::map<int, float>{});
    check(std::map<int, double>{});
}

TEST_F(autoclass, mapping_setitem) {
    auto check = [](auto type) {
        using M = decltype(type);
        using key_type = typename M::key_type;
        using value_type = typename M::mapped_type;

        py::scoped_ref cls = py::autoclass<M>()
                                 .template new_<>()
                                 .template mapping<key_type, value_type>()
                                 .type();
        ASSERT_TRUE(cls);

        py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls));
        ASSERT_TRUE(inst);
        ASSERT_EQ(Py_TYPE(inst.get()), cls.get());
        M& unboxed = py::autoclass<M>::unbox(inst);

        {
            // prepopulate 1 to test an overwrite
            key_type key = 1;
            value_type old_value = 1;
            value_type new_value = 2;
            unboxed[key] = old_value;

            py::scoped_ref boxed_key = py::to_object(key);
            ASSERT_TRUE(boxed_key);
            py::scoped_ref boxed_value = py::to_object(new_value);
            ASSERT_TRUE(boxed_value);
            ASSERT_FALSE(
                PyObject_SetItem(inst.get(), boxed_key.get(), boxed_value.get()));
            py::scoped_ref result(PyObject_GetItem(inst.get(), boxed_key.get()));
            ASSERT_TRUE(result);
            value_type unboxed_value = py::from_object<value_type>(result);
            EXPECT_EQ(unboxed_value, new_value);
        }

        {
            // test adding a new key
            key_type key = 2;
            value_type value = 3;

            py::scoped_ref boxed_key = py::to_object(key);
            ASSERT_TRUE(boxed_key);
            py::scoped_ref boxed_value = py::to_object(value);
            ASSERT_TRUE(boxed_value);
            ASSERT_FALSE(
                PyObject_SetItem(inst.get(), boxed_key.get(), boxed_value.get()));
            py::scoped_ref result(PyObject_GetItem(inst.get(), boxed_key.get()));
            ASSERT_TRUE(result);
            value_type unboxed_value = py::from_object<value_type>(result);
            EXPECT_EQ(unboxed_value, value);
        }
    };

    check(std::map<int, int>{});
    check(std::map<char, int>{});
    check(std::map<int, float>{});
    check(std::map<int, double>{});
}

TEST_F(autoclass, mapping_throws) {
    struct M {
        [[noreturn]] int& operator[](std::size_t) {
            throw std::range_error("ix is out of range");
        }
    };

    py::scoped_ref cls = py::autoclass<M>().new_<>().mapping<std::size_t, int>().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls));
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    py::scoped_ref key = py::to_object(10);
    ASSERT_TRUE(key);
    py::scoped_ref boxed_result(PyObject_GetItem(inst.get(), key.get()));
    EXPECT_FALSE(boxed_result);
    expect_pyerr_type_and_message(PyExc_RuntimeError,
                                  "a C++ exception was raised: ix is out of range");
    PyErr_Clear();

    py::scoped_ref value = py::to_object(10);
    ASSERT_TRUE(value);
    ASSERT_TRUE(PyObject_SetItem(inst.get(), key.get(), value.get()));
    expect_pyerr_type_and_message(PyExc_RuntimeError,
                                  "a C++ exception was raised: ix is out of range");
    PyErr_Clear();
}

TEST_F(autoclass, iter) {
    using T = std::vector<int>;

    py::scoped_ref cls = py::autoclass<T>().new_<int>().iter().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls), 5);
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    T& unboxed = py::autoclass<T>::unbox(inst);
    ASSERT_EQ(unboxed.size(), 5ul);
    std::iota(unboxed.begin(), unboxed.end(), 0);

    Py_ssize_t starting_ref = Py_REFCNT(inst.get());
    {
        py::scoped_ref it(PyObject_GetIter(inst.get()));
        EXPECT_EQ(Py_REFCNT(inst.get()), starting_ref + 1);
    }
    EXPECT_EQ(Py_REFCNT(inst.get()), starting_ref);

    py::scoped_ref fast_seq(PySequence_Fast(inst.get(), "expected inst to be iterable"));
    ASSERT_TRUE(fast_seq);

    ASSERT_EQ(PySequence_Fast_GET_SIZE(fast_seq.get()), 5);
    for (int ix = 0; ix < 5; ++ix) {
        int unboxed = py::from_object<int>(PySequence_Fast_GET_ITEM(fast_seq.get(), ix));
        EXPECT_EQ(unboxed, ix);
    }
}

TEST_F(autoclass, iter_throws) {
    struct C {
        struct iterator {
            int ix;

            [[noreturn]] int& operator*() const {
                throw std::runtime_error{"ayy"};
            }

            iterator& operator++() {
                ++ix;
                return *this;
            }

            bool operator!=(const iterator& other) const {
                return ix != other.ix;
            }
        };

        iterator begin() {
            return {0};
        }

        iterator end() {
            return {1};
        }
    };

    py::scoped_ref cls = py::autoclass<C>().new_<>().iter().type();
    ASSERT_TRUE(cls);

    py::scoped_ref inst = py::call_function(static_cast<PyObject*>(cls));
    ASSERT_TRUE(inst);
    ASSERT_EQ(Py_TYPE(inst.get()), cls.get());

    py::scoped_ref fast_seq(PySequence_Fast(inst.get(), "expected inst to be iterable"));
    EXPECT_FALSE(fast_seq);
    expect_pyerr_type_and_message(PyExc_RuntimeError, "a C++ exception was raised: ayy");
    PyErr_Clear();
}
#endif
}  // namespace test_autoclass
