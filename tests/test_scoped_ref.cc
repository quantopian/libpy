#include "Python.h"
#include "gtest/gtest.h"

#include "libpy/scoped_ref.h"
#include "test_utils.h"

namespace test_scoped_ref {
class scoped_ref : public with_python_interpreter {};

TEST_F(scoped_ref, basic_lifetime) {
    PyObject* raw = Py_None;
    auto starting_ref_count = Py_REFCNT(raw);

    {
        Py_INCREF(raw);

        // wrapping an object in a scoped ref claims the reference, it should not incref
        // again
        auto sr = py::scoped_ref(raw);

        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);
    }

    EXPECT_EQ(Py_REFCNT(raw), starting_ref_count);
}

TEST_F(scoped_ref, copy_construct) {
    PyObject* raw = Py_None;
    auto starting_ref_count = Py_REFCNT(raw);

    {
        Py_INCREF(raw);
        auto sr = py::scoped_ref(raw);

        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);

        EXPECT_EQ(sr.get(), raw);

        {
            // copy construct
            py::scoped_ref<> copy(sr);

            // copy should take new ownership of the underlying object
            EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 2);

            EXPECT_EQ(copy, sr);
            EXPECT_EQ(copy.get(), sr.get());
        }

        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);
    }

    EXPECT_EQ(Py_REFCNT(raw), starting_ref_count);
}

TEST_F(scoped_ref, self_assign) {
    PyObject* raw = Py_None;
    Py_INCREF(raw);

    auto starting_ref_count = Py_REFCNT(raw);
    auto ref = py::scoped_ref(raw);
    ref = ref;
    auto ending_ref_count = Py_REFCNT(raw);
    EXPECT_EQ(ending_ref_count, starting_ref_count);
}

TEST_F(scoped_ref, assign_same_underlying_pointer) {
    PyObject* raw = Py_None;
    auto start_ref_count = Py_REFCNT(raw);

    {
        Py_INCREF(raw);
        py::scoped_ref a(raw);
        EXPECT_EQ(Py_REFCNT(raw), start_ref_count + 1);

        Py_INCREF(raw);
        py::scoped_ref b(raw);
        EXPECT_EQ(Py_REFCNT(raw), start_ref_count + 2);

        a = b;

        EXPECT_EQ(Py_REFCNT(raw), start_ref_count + 2);
    }

    EXPECT_EQ(Py_REFCNT(raw), start_ref_count);
}

TEST_F(scoped_ref, move_construct) {
    PyObject* raw = Py_None;
    auto starting_ref_count = Py_REFCNT(raw);

    {
        Py_INCREF(raw);
        auto sr = py::scoped_ref(raw);
        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);

        {
            // move construct
            py::scoped_ref<> moved(std::move(sr));

            EXPECT_EQ(moved.get(), raw);

            // movement doesn't alter the refcount of the underlying
            EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);

            // movement resets the moved-from scoped ref
            EXPECT_EQ(sr.get(), nullptr);
        }

        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count);
    }

    EXPECT_EQ(Py_REFCNT(raw), starting_ref_count);
}

TEST_F(scoped_ref, move_assign) {
    PyObject* raw_rhs = Py_None;
    auto rhs_starting_ref_count = Py_REFCNT(raw_rhs);

    PyObject* raw_lhs = Py_Ellipsis;
    auto lhs_starting_ref_count = Py_REFCNT(raw_lhs);

    {
        Py_INCREF(raw_rhs);
        py::scoped_ref rhs(raw_rhs);
        EXPECT_EQ(Py_REFCNT(raw_rhs), rhs_starting_ref_count + 1);

        {
            Py_INCREF(raw_lhs);
            py::scoped_ref lhs(raw_lhs);
            EXPECT_EQ(Py_REFCNT(raw_lhs), lhs_starting_ref_count + 1);

            lhs = std::move(rhs);

            EXPECT_EQ(lhs.get(), raw_rhs);

            // movement doesn't alter the refcount of either underlying object
            EXPECT_EQ(Py_REFCNT(raw_lhs), lhs_starting_ref_count + 1);
            EXPECT_EQ(Py_REFCNT(raw_rhs), rhs_starting_ref_count + 1);

            // move assign swaps the values to ensure the old value is cleaned up
            EXPECT_EQ(rhs.get(), raw_lhs);
        }

        // rhs was cleaned up in the previous scope
        EXPECT_EQ(Py_REFCNT(raw_rhs), rhs_starting_ref_count);

        // lhs has been extended to the lifetime of this outer scope
        EXPECT_EQ(Py_REFCNT(raw_lhs), lhs_starting_ref_count + 1);
    }

    EXPECT_EQ(Py_REFCNT(raw_rhs), rhs_starting_ref_count);
    EXPECT_EQ(Py_REFCNT(raw_lhs), lhs_starting_ref_count);
}

TEST_F(scoped_ref, escape) {
    PyObject* raw = Py_None;
    auto starting_ref_count = Py_REFCNT(raw);

    PyObject* escape_into = nullptr;

    {
        Py_INCREF(raw);
        auto sr = py::scoped_ref(raw);
        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);

        escape_into = std::move(sr).escape();

        // escaping from a scoped ref should reset the pointer
        EXPECT_EQ(sr.get(), nullptr);
    }

    EXPECT_EQ(escape_into, raw);

    // `sr` was moved from, it shouldn't decref the object
    EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);

    // keep the test refcount neutral
    Py_DECREF(raw);
}

TEST_F(scoped_ref, operator_bool) {
    PyObject* raw = Py_None;

    Py_INCREF(raw);
    py::scoped_ref truthy(raw);
    EXPECT_TRUE(truthy);

    py::scoped_ref falsy(nullptr);
    EXPECT_FALSE(falsy);
}
}  // namespace test_scoped_ref
