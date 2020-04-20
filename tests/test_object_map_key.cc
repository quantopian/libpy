#include <map>
#include <unordered_map>

#include "gtest/gtest.h"

#include "libpy/dense_hash_map.h"
#include "libpy/exception.h"
#include "libpy/meta.h"
#include "libpy/object_map_key.h"
#include "libpy/owned_ref.h"
#include "libpy/to_object.h"
#include "test_utils.h"

namespace test_object_hash_key {
class object_map_key : public with_python_interpreter {};

TEST_F(object_map_key, eq) {
    py::object_map_key a{py::to_object(9001)};
    ASSERT_TRUE(a);

    py::object_map_key b{py::to_object(9001)};
    ASSERT_TRUE(b);

    // check that we aren't just checking pointer equality
    EXPECT_NE(a.get(), b.get());
    EXPECT_EQ(a, b);

    EXPECT_EQ(py::object_map_key{nullptr}, py::object_map_key{nullptr});
}

TEST_F(object_map_key, ne) {
    py::object_map_key a{py::to_object(9001)};
    ASSERT_TRUE(a);

    py::object_map_key b{py::to_object(9002)};
    ASSERT_TRUE(b);

    // check that we aren't just checking pointer equality
    EXPECT_NE(a.get(), b.get());
    EXPECT_NE(a, b);
    EXPECT_NE(a, nullptr);
}

TEST_F(object_map_key, lt) {
    py::object_map_key a{py::to_object(9000)};
    ASSERT_TRUE(a);

    py::object_map_key b{py::to_object(9001)};
    ASSERT_TRUE(b);

    // check that we aren't just checking pointer equality
    EXPECT_NE(a.get(), b.get());
    EXPECT_LT(a, b);

    EXPECT_LT(a, nullptr);

    // NOTE: don't test EXPECT_GE` here because we are testing `operator<` explicitly
    EXPECT_FALSE(py::object_map_key{nullptr} < a);
    EXPECT_FALSE(py::object_map_key{nullptr} < nullptr);
}

TEST_F(object_map_key, le) {
    py::object_map_key a{py::to_object(9000)};
    ASSERT_TRUE(a);

    py::object_map_key b{py::to_object(9001)};
    ASSERT_TRUE(b);

    py::object_map_key c{py::to_object(9001)};
    ASSERT_TRUE(c);

    // check that we aren't just checking pointer equality
    EXPECT_NE(a.get(), b.get());
    EXPECT_LE(a, b);

    EXPECT_NE(a.get(), c.get());
    EXPECT_LE(a, c);

    EXPECT_LE(a, nullptr);

    // NOTE: don't test EXPECT_GE` here because we are testing `operator<` explicitly
    EXPECT_FALSE(py::object_map_key{nullptr} <= a);
    EXPECT_LE(py::object_map_key{nullptr}, nullptr);
}

TEST_F(object_map_key, ge) {
    py::object_map_key a{py::to_object(9000)};
    ASSERT_TRUE(a);

    py::object_map_key b{py::to_object(9001)};
    ASSERT_TRUE(b);

    py::object_map_key c{py::to_object(9001)};
    ASSERT_TRUE(c);

    // check that we aren't just checking pointer equality
    EXPECT_NE(a.get(), b.get());
    EXPECT_GE(b, a);

    EXPECT_NE(a.get(), c.get());
    EXPECT_GE(b, c);

    // NOTE: don't test EXPECT_LT` here because we are testing `operator>=` explicitly
    EXPECT_FALSE(a >= nullptr);
    EXPECT_GE(py::object_map_key{nullptr}, a);
    EXPECT_GE(py::object_map_key{nullptr}, nullptr);
}

TEST_F(object_map_key, gt) {
    py::object_map_key a{py::to_object(9000)};
    ASSERT_TRUE(a);

    py::object_map_key b{py::to_object(9001)};
    ASSERT_TRUE(b);

    // check that we aren't just checking pointer equality
    EXPECT_NE(a.get(), b.get());
    EXPECT_GT(b, a);

    // NOTE: don't test EXPECT_LE` here because we are testing `operator>` explicitly
    EXPECT_FALSE(a > nullptr);
    EXPECT_FALSE(py::object_map_key{nullptr} < nullptr);
    EXPECT_GT(py::object_map_key{nullptr}, a);
}

template<typename F>
void test_fails(const std::string& method, F f) {
    using namespace std::literals;

    py::owned_ref ns = RUN_PYTHON(R"(
        class C(object):
            def __)"s + method + R"(__(self, other):
                raise ValueError('ayy lmao')

        a = C()
        b = C()
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref a_ob = PyDict_GetItemString(ns.get(), "a");
    ASSERT_TRUE(a_ob);
    Py_INCREF(a_ob);
    py::object_map_key a{a_ob};

    py::borrowed_ref b_ob = PyDict_GetItemString(ns.get(), "b");
    ASSERT_TRUE(b_ob);
    Py_INCREF(b_ob);
    py::object_map_key b{b_ob};

    EXPECT_THROW(static_cast<void>(f(a, b)), py::exception);
    expect_pyerr_type_and_message(PyExc_ValueError, "ayy lmao");
    PyErr_Clear();
}

TEST_F(object_map_key, eq_fails) {
    test_fails("eq", py::meta::op::eq{});
}

TEST_F(object_map_key, ne_fails) {
    test_fails("ne", py::meta::op::ne{});
}

TEST_F(object_map_key, lt_fails) {
    test_fails("lt", py::meta::op::lt{});
}

TEST_F(object_map_key, le_fails) {
    test_fails("le", py::meta::op::le{});
}

TEST_F(object_map_key, ge_fails) {
    test_fails("ge", py::meta::op::ge{});
}

TEST_F(object_map_key, gt_fails) {
    test_fails("gt", py::meta::op::gt{});
}

TEST_F(object_map_key, hash) {
    int value = 10;
    py::object_map_key a{py::to_object(value)};
    ASSERT_TRUE(a);

    EXPECT_EQ(std::hash<py::object_map_key>{}(a), value);

    // nullptr just hashes to 0
    EXPECT_EQ(std::hash<py::object_map_key>{}(py::object_map_key{}), 0);
}

TEST_F(object_map_key, hash_fails) {
    py::owned_ref ns = RUN_PYTHON(R"(
        class C(object):
            def __hash__(self, other):
                raise ValueError()

        a = C()
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref a_ob = PyDict_GetItemString(ns.get(), "a");
    ASSERT_TRUE(a_ob);
    Py_INCREF(a_ob);
    py::object_map_key a{a_ob};

    EXPECT_THROW(static_cast<void>(std::hash<py::object_map_key>{}(a)), py::exception);
    PyErr_Clear();
}

template<typename M>
void test_use_in_map(M map) {

    int a_value = 10;
    py::object_map_key a_key{py::to_object(a_value)};
    ASSERT_TRUE(a_key);

    map[a_key] = a_value;

    EXPECT_EQ(map[a_key], a_value);

    int b_value = 20;
    // use a scoped ref to test implicit conversions
    py::owned_ref b_key = py::to_object(b_value);
    ASSERT_TRUE(b_key);

    map[b_key] = b_value;

    EXPECT_EQ(map[b_key], b_value);
    EXPECT_EQ(map[a_key], a_value);

    int c_value = 30;
    py::owned_ref c_key = py::to_object(c_value);
    ASSERT_TRUE(c_key);

    map.insert({c_key, c_value});

    EXPECT_EQ(map[c_key], c_value);
    EXPECT_EQ(map[b_key], b_value);
    EXPECT_EQ(map[a_key], a_value);
}

TEST_F(object_map_key, use_in_map) {
    test_use_in_map(std::map<py::object_map_key, int>{});
    test_use_in_map(std::unordered_map<py::object_map_key, int>{});
    test_use_in_map(py::sparse_hash_map<py::object_map_key, int>{});
    test_use_in_map(py::dense_hash_map<py::object_map_key, int>{py::object_map_key{}});
}

TEST_F(object_map_key, convert) {
    py::owned_ref<> ob = py::to_object(1);
    ASSERT_TRUE(ob);

    py::object_map_key m = ob;
    EXPECT_EQ(m.get(), ob.get());
    EXPECT_EQ(static_cast<py::owned_ref<>>(m).get(), ob.get());
}
}  // namespace test_object_hash_key
