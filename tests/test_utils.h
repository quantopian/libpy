#pragma once

#include <string>
#include <sstream>

#include "Python.h"
#include "gtest/gtest.h"

class with_python_interpreter : public testing::Test {
private:
    std::string format_exception(PyObject* type, PyObject* value, PyObject*) {
        std::stringstream buf;

        if (!type) {
            buf << "<unknown type>";
        }
        else {
            buf << reinterpret_cast<PyTypeObject*>(type)->tp_name;
        }

        buf << ": ";

        if (!value) {
            buf << "<nullptr>";
        }
        else {
            PyObject* as_string = PyObject_ASCII(value);
            if (!as_string) {
                buf << "<nullptr>";
            }
            else {
                buf << PyUnicode_AsUTF8(as_string);
                Py_DECREF(as_string);
            }
        }

        return buf.str();
    }
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
        PyObject* type;
        PyObject* value;
        PyObject* traceback;
        PyErr_Fetch(&type, &value, &traceback);

        EXPECT_FALSE(type || value || traceback)
            << format_exception(type, value, traceback);
    }
};
