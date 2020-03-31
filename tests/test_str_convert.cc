#include "gtest/gtest.h"

#include "libpy/str_convert.h"
#include "test_utils.h"

namespace test_str_convert {

using namespace py::cs::literals;

class to_stringlike : public with_python_interpreter {};

TEST_F(to_stringlike, bytes) {
    auto s = "foobar"_cs;
    const char* expected = "foobar";

    py::owned_ref<> s_py = py::to_stringlike(s, py::str_type::bytes);
    ASSERT_TRUE(s_py);

    ASSERT_TRUE(PyBytes_CheckExact(s_py.get()));
    EXPECT_STREQ(PyBytes_AS_STRING(s_py.get()), expected);
}

TEST_F(to_stringlike, str) {
    auto s = "foobar"_cs;
    const char* expected = "foobar";

    py::owned_ref<> s_py = py::to_stringlike(s, py::str_type::str);

#if PY_MAJOR_VERSION == 2
    ASSERT_TRUE(PyString_CheckExact(s_py.get()));
    ASSERT_STREQ(PyString_AS_STRING(s_py.get()), expected);
#else
    ASSERT_TRUE(PyUnicode_CheckExact(s_py.get()));
    py::owned_ref<> decoded(PyUnicode_AsEncodedString(s_py.get(), "utf-8", "strict"));
    ASSERT_TRUE(decoded);
    EXPECT_STREQ(PyBytes_AS_STRING(decoded.get()), expected);
#endif
}

TEST_F(to_stringlike, unicode) {
    auto s = "foobar"_cs;
    const char* expected = "foobar";

    py::owned_ref<> s_py = py::to_stringlike(s, py::str_type::unicode);

    ASSERT_TRUE(PyUnicode_CheckExact(s_py.get()));
    py::owned_ref<> decoded(PyUnicode_AsEncodedString(s_py.get(), "utf-8", "strict"));
    ASSERT_TRUE(decoded);
    EXPECT_STREQ(PyBytes_AS_STRING(decoded.get()), expected);
}

}  // namespace test_str_convert
