#pragma once

#include <forward_list>
#include <unordered_map>
#include <vector>

#include "libpy/abi.h"
#include "libpy/borrowed_ref.h"
#include "libpy/detail/api.h"
#include "libpy/detail/numpy.h"
#include "libpy/detail/python.h"
#include "libpy/owned_ref.h"

#define _libpy_XSTR(s) #s
#define _libpy_STR(s) _libpy_XSTR(s)

#define _libpy_MODULE_PATH(parent, name) _libpy_STR(parent) "." _libpy_STR(name)

#define _libpy_XCAT(a, b) a##b
#define _libpy_CAT(a, b) _libpy_XCAT(a, b)

#define _libpy_MODINIT_NAME(name) _libpy_CAT(PyInit_, name)
#define _libpy_MODULE_CREATE(path) PyModule_Create(&_libpy_module)

/** Define a Python module.

    @param parent A symbol indicating the parent module.
    @param name The leaf name of the module.
    @param methods ({...}) list of  objects representing the functions to add to the
                    module. Note this list must be surrounded by parentheses.

    ## Examples

    Create a module `my_package.submodule.my_module` with two functions `f` and
    `g` and one type `T`.

    \code
    LIBPY_AUTOMODULE(my_package.submodule,
                     my_module,
                     ({py::autofunction<f>("f"),
                       py::autofunction<g>("g")}))
    (py::borrowed_ref<> m) {
        py::borrowed_ref t = py::autoclass<T>("T").new_().type();
        return PyObject_SetAttrString(m.get(), "T", static_cast<PyObject*>(t));
    }
    /endcode
 */
#define LIBPY_AUTOMODULE(parent, name, methods)                                          \
    bool _libpy_user_mod_init(py::borrowed_ref<>);                                       \
    PyMODINIT_FUNC _libpy_MODINIT_NAME(name)() LIBPY_EXPORT;                             \
    PyMODINIT_FUNC _libpy_MODINIT_NAME(name)() {                                         \
        import_array();                                                                  \
        if (py::abi::ensure_compatible_libpy_abi()) {                                    \
            return nullptr;                                                              \
        }                                                                                \
        static std::vector<PyMethodDef> ms methods;                                      \
        ms.emplace_back(py::end_method_list);                                            \
        static PyModuleDef _libpy_module{                                                \
            PyModuleDef_HEAD_INIT,                                                       \
            _libpy_MODULE_PATH(parent, name),                                            \
            nullptr,                                                                     \
            -1,                                                                          \
            ms.data(),                                                                   \
        };                                                                               \
        py::owned_ref m(_libpy_MODULE_CREATE(_libpy_MODULE_PATH(parent, name)));         \
        if (!m) {                                                                        \
            return nullptr;                                                              \
        }                                                                                \
        try {                                                                            \
            if (_libpy_user_mod_init(m)) {                                               \
                return nullptr;                                                          \
            }                                                                            \
        }                                                                                \
        catch (const std::exception& e) {                                                \
            py::raise_from_cxx_exception(e);                                             \
            return nullptr;                                                              \
        }                                                                                \
        catch (...) {                                                                    \
            if (!PyErr_Occurred()) {                                                     \
                py::raise(PyExc_RuntimeError) << "an unknown C++ exception was raised";  \
                return nullptr;                                                          \
            }                                                                            \
        }                                                                                \
        return std::move(m).escape();                                                    \
    }                                                                                    \
    bool _libpy_user_mod_init
