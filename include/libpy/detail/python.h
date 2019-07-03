#pragma once

// This triggers a warning in Python 2, add a standard place to find this header that
// doesn't trigger this warning.
#if PY_MAJOR_VERSION == 2
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE
#endif
#include <Python.h>
