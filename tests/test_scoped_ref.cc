#include "gtest/gtest.h"

#include "libpy/call_function.h"
#include "libpy/detail/python.h"
#include "libpy/owned_ref.h"
#include "test_utils.h"

namespace test_owned_ref {
class owned_ref : public with_python_interpreter {};

TEST_F(owned_ref, basic_lifetime) {
    PyObject* raw = Py_None;
    auto starting_ref_count = Py_REFCNT(raw);

    {
        Py_INCREF(raw);

        // wrapping an object in a scoped ref claims the reference, it should not
        // incref
        // again
        py::owned_ref sr(raw);

        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);
    }

    EXPECT_EQ(Py_REFCNT(raw), starting_ref_count);
}

TEST_F(owned_ref, copy_construct) {
    PyObject* raw = Py_None;
    auto starting_ref_count = Py_REFCNT(raw);

    {
        Py_INCREF(raw);
        py::owned_ref sr(raw);

        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);

        EXPECT_EQ(sr.get(), raw);

        {
            // copy construct
            py::owned_ref<> copy(sr);

            // copy should take new ownership of the underlying object
            EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 2);

            EXPECT_EQ(copy, sr);
            EXPECT_EQ(copy.get(), sr.get());
        }

        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);
    }

    EXPECT_EQ(Py_REFCNT(raw), starting_ref_count);
}

TEST_F(owned_ref, self_assign) {
    py::owned_ref ob = nullptr;
    // we need to hold onto the ref when the namespace goes out of scope so that
    // the callback will still fire
    py::owned_ref ref = nullptr;
    py::owned_ref destructions = nullptr;
    {
        py::owned_ref ns = RUN_PYTHON(R"(
            import weakref

            class EmptyObject(object):
                pass

            def setup():
                ob = EmptyObject()

                # use a list so we can see changes to this object without a cell
                destructions = []

                def callback(wr):
                    destructions.append(True)

                return ob, weakref.ref(ob, callback), destructions
        )");
        ASSERT_TRUE(ns);
        ASSERT_TRUE(PyDict_CheckExact(ns.get()));

        PyObject* setup = PyDict_GetItemString(ns.get(), "setup");
        ASSERT_TRUE(setup);

        auto res = py::call_function(setup);
        ASSERT_TRUE(res);
        ASSERT_TRUE(PyTuple_CheckExact(res.get()));
        ASSERT_EQ(PyTuple_GET_SIZE(res.get()), 3);

        ob = py::owned_ref(PyTuple_GET_ITEM(res.get(), 0));
        Py_INCREF(ob);

        ref = py::owned_ref(PyTuple_GET_ITEM(res.get(), 1));
        Py_INCREF(ref);

        destructions = py::owned_ref(PyTuple_GET_ITEM(res.get(), 2));
        Py_INCREF(destructions);
        ASSERT_TRUE(PyList_CheckExact(destructions.get()));
    }

    auto starting_ref_count = Py_REFCNT(ob);
    EXPECT_EQ(starting_ref_count, 1);

    ob = ob;
    auto ending_ref_count = Py_REFCNT(ob);
    EXPECT_EQ(ending_ref_count, starting_ref_count);
    EXPECT_FALSE(PyList_GET_SIZE(destructions.get()));

    // explicitly kill ob now
    PyObject* escaped = std::move(ob).escape();
    Py_DECREF(escaped);

    // make sure the callback fired
    EXPECT_EQ(PyList_GET_SIZE(destructions.get()), 1);
}

TEST_F(owned_ref, assign_same_underlying_pointer) {
    PyObject* raw = Py_None;
    auto start_ref_count = Py_REFCNT(raw);

    {
        Py_INCREF(raw);
        py::owned_ref a(raw);
        EXPECT_EQ(Py_REFCNT(raw), start_ref_count + 1);

        Py_INCREF(raw);
        py::owned_ref b(raw);
        EXPECT_EQ(Py_REFCNT(raw), start_ref_count + 2);

        a = b;

        EXPECT_EQ(Py_REFCNT(raw), start_ref_count + 2);
    }

    EXPECT_EQ(Py_REFCNT(raw), start_ref_count);
}

TEST_F(owned_ref, move_construct) {
    PyObject* raw = Py_None;
    auto starting_ref_count = Py_REFCNT(raw);

    {
        Py_INCREF(raw);
        py::owned_ref sr(raw);
        EXPECT_EQ(Py_REFCNT(raw), starting_ref_count + 1);

        {
            // move construct
            py::owned_ref moved(std::move(sr));

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

TEST_F(owned_ref, move_assign) {
    PyObject* raw_rhs = Py_None;
    auto rhs_starting_ref_count = Py_REFCNT(raw_rhs);

    PyObject* raw_lhs = Py_Ellipsis;
    auto lhs_starting_ref_count = Py_REFCNT(raw_lhs);

    {
        Py_INCREF(raw_rhs);
        py::owned_ref rhs(raw_rhs);
        EXPECT_EQ(Py_REFCNT(raw_rhs), rhs_starting_ref_count + 1);

        {
            Py_INCREF(raw_lhs);
            py::owned_ref lhs(raw_lhs);
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

TEST_F(owned_ref, escape) {
    PyObject* raw = Py_None;
    auto starting_ref_count = Py_REFCNT(raw);

    PyObject* escape_into = nullptr;

    {
        Py_INCREF(raw);
        py::owned_ref sr(raw);
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

TEST_F(owned_ref, operator_bool) {
    PyObject* raw = Py_None;

    Py_INCREF(raw);
    py::owned_ref truthy(raw);
    EXPECT_TRUE(truthy);

    py::owned_ref falsy(nullptr);
    EXPECT_FALSE(falsy);
}
}  // namespace test_owned_ref
