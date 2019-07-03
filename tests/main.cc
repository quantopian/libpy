#include "gtest/gtest.h"

#include "libpy/detail/python.h"
#define LIBPY_TEST_MAIN
#include "libpy/numpy_utils.h"

int main(int argc, char** argv) {
    Py_Initialize();
    py::ensure_import_array();

    testing::InitGoogleTest(&argc, argv);
    int out = RUN_ALL_TESTS();
    Py_Finalize();
    return out;
}
