#include "Python.h"
#include "gtest/gtest.h"

#include "libpy/scoped_ref.h"
#include "test_utils.h"

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

TEST_F(scoped_ref, copy) {
    PyObject* raw = Py_None;
    auto starting_ref_count = Py_REFCNT(raw);

    {
        Py_INCREF(raw);
        auto sr = py::scoped_ref(raw);

        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);

        EXPECT_EQ(sr.get(), raw);

        {
            // copy construct
            py::scoped_ref<PyObject> copy = sr;

            // copy should take new ownership of the underlying object
            EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 2);

            EXPECT_EQ(copy, sr);
            EXPECT_EQ(copy.get(), sr.get());
        }

        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);
    }

    EXPECT_EQ(Py_REFCNT(raw), starting_ref_count);
}

TEST_F(scoped_ref, move) {
    PyObject* raw = Py_None;
    auto starting_ref_count = Py_REFCNT(raw);

    {
        Py_INCREF(raw);
        auto sr = py::scoped_ref(raw);
        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);

        {
            // move construct
            py::scoped_ref<PyObject> moved = std::move(sr);

            // movement doesn't alter the refcount of the underlying
            EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);

            EXPECT_EQ(moved.get(), raw);

            // movement resets the moved-from scoped ref
            EXPECT_EQ(sr.get(), nullptr);
        }

        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count);
    }

    EXPECT_EQ(Py_REFCNT(raw), starting_ref_count);
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
    py::scoped_ref<PyObject> truthy(raw);
    EXPECT_TRUE(truthy);

    py::scoped_ref<PyObject> falsy(nullptr);
    EXPECT_FALSE(falsy);
}
