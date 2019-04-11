#include "gtest/gtest.h"
#include <Python.h>

#define LIBPY_TEST_MAIN
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_libpy
#include "libpy/numpy_utils.h"

int main(int argc, char** argv) {
    Py_Initialize();
    py::ensure_import_array();

    testing::InitGoogleTest(&argc, argv);
    int out =  RUN_ALL_TESTS();
    Py_Finalize();
    return out;
}
