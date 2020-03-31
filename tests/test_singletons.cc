#include "gtest/gtest.h"

#include "libpy/singletons.h"

namespace test_singletons {
TEST(none, is_none) {
    EXPECT_EQ(py::none, py::none);
    EXPECT_EQ(py::none.get(), Py_None);
}

TEST(none, refcnt) {
    Py_ssize_t start = Py_REFCNT(Py_None);
    {
        py::owned_ref<> none = py::none;
        EXPECT_EQ(Py_REFCNT(Py_None), start + 1);
    }
    EXPECT_EQ(Py_REFCNT(Py_None), start);
}

TEST(ellipsis, is_ellipsis) {
    EXPECT_EQ(py::ellipsis, py::ellipsis);
    EXPECT_EQ(py::ellipsis.get(), Py_Ellipsis);
}

TEST(ellipsis, refcnt) {
    Py_ssize_t start = Py_REFCNT(Py_Ellipsis);
    {
        py::owned_ref<> ellipsis = py::ellipsis;
        EXPECT_EQ(Py_REFCNT(Py_Ellipsis), start + 1);
    }
    EXPECT_EQ(Py_REFCNT(Py_Ellipsis), start);
}

TEST(not_implemented, is_not_implemented) {
    EXPECT_EQ(py::not_implemented, py::not_implemented);
    EXPECT_EQ(py::not_implemented.get(), Py_NotImplemented);
}

TEST(not_implemented, refcnt) {
    Py_ssize_t start = Py_REFCNT(Py_NotImplemented);
    {
        py::owned_ref<> not_implemented = py::not_implemented;
        EXPECT_EQ(Py_REFCNT(Py_NotImplemented), start + 1);
    }
    EXPECT_EQ(Py_REFCNT(Py_NotImplemented), start);
}
}  // namespace test_singletons
