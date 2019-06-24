#include <Python.h>

#if PY_MAJOR_VERSION == 2
#define Py_RETURN_NOTIMPLEMENTED \
    Py_INCREF(Py_NotImplemented); \
    return Py_NotImplemented;

using Py_hash_t = long;
#endif
