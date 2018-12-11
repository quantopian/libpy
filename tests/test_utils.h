#pragma once

#include "Python.h"
#include "gtest/gtest.h"

class with_python_interpreter : public testing::Test {
public:
    static void SetUpTestCase() {
        // initialize the Python interpreter state
        Py_Initialize();
    }

    static void TearDownTestCase() {
        // tear down the Python interpreter state
        Py_Finalize();
    }

    virtual void TearDown() {
        EXPECT_FALSE(PyErr_Occurred()) << "test ended with Python exception set";
        PyErr_Clear();
    }
};
