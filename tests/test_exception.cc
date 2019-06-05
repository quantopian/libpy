#include "libpy/exception.h"
#include "test_utils.h"

namespace test_exception {
class exception : public with_python_interpreter {};

void assert_pyerr_type_and_message(PyObject* ptype, std::string_view pmsg) {
    EXPECT_TRUE(PyErr_ExceptionMatches(ptype));

    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_Restore(type, value, traceback);

    py::scoped_ref py_msg(PyObject_Str(value));
    ASSERT_TRUE(py_msg);
    std::string_view c_msg = py::util::pystring_to_string_view(py_msg);
    EXPECT_EQ(c_msg, pmsg);
}

TEST_F(exception, raise_from_cxx) {
    ASSERT_FALSE(PyErr_Occurred());

    py::raise_from_cxx_exception(std::runtime_error("msg"));
    assert_pyerr_type_and_message(PyExc_RuntimeError, "a C++ exception was raised: msg");

    // Raising again should preserve existing error indicator type and append to the
    // message
    py::raise_from_cxx_exception(std::runtime_error("msg2"));
    assert_pyerr_type_and_message(
        PyExc_RuntimeError,
        "a C++ exception was raised: msg (raised from C++ exception: msg2)");

    PyErr_Clear();

    PyErr_SetString(PyExc_IndentationError, "pymsg");
    py::raise_from_cxx_exception(std::runtime_error("msg"));
    assert_pyerr_type_and_message(PyExc_IndentationError,
                                  "pymsg (raised from C++ exception: msg)");

    // Raising again should preserve existing error indicator type and append to the
    // message
    py::raise_from_cxx_exception(std::runtime_error("msg2"));
    assert_pyerr_type_and_message(
        PyExc_IndentationError,
        "pymsg (raised from C++ exception: msg) (raised from C++ exception: msg2)");

    PyErr_Clear();
}
}  // namespace test_exception
