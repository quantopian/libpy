#pragma once

#include <cstddef>

#include <Python.h>

#include "libpy/py2.h"

namespace py {
#if PY_MAJOR_VERSION == 2

constexpr int Py_mp_ass_subscript = 3;
constexpr int Py_mp_length = 4;
constexpr int Py_mp_subscript = 5;
constexpr int Py_nb_absolute = 6;
constexpr int Py_nb_add = 7;
constexpr int Py_nb_and = 8;
constexpr int Py_nb_bool = 9;  // this is nb_nonzero in Python 2
constexpr int Py_nb_divmod = 10;
constexpr int Py_nb_float = 11;
constexpr int Py_nb_floor_divide = 12;
constexpr int Py_nb_index = 13;
constexpr int Py_nb_inplace_add = 14;
constexpr int Py_nb_inplace_and = 15;
constexpr int Py_nb_inplace_floor_divide = 16;
constexpr int Py_nb_inplace_lshift = 17;
constexpr int Py_nb_inplace_multiply = 18;
constexpr int Py_nb_inplace_or = 19;
constexpr int Py_nb_inplace_power = 20;
constexpr int Py_nb_inplace_remainder = 21;
constexpr int Py_nb_inplace_rshift = 22;
constexpr int Py_nb_inplace_subtract = 23;
constexpr int Py_nb_inplace_true_divide = 24;
constexpr int Py_nb_inplace_xor = 25;
constexpr int Py_nb_int = 26;
constexpr int Py_nb_invert = 27;
constexpr int Py_nb_lshift = 28;
constexpr int Py_nb_multiply = 29;
constexpr int Py_nb_negative = 30;
constexpr int Py_nb_or = 31;
constexpr int Py_nb_positive = 32;
constexpr int Py_nb_power = 33;
constexpr int Py_nb_remainder = 34;
constexpr int Py_nb_rshift = 35;
constexpr int Py_nb_subtract = 36;
constexpr int Py_nb_true_divide = 37;
constexpr int Py_nb_xor = 38;
constexpr int Py_sq_ass_item = 39;
constexpr int Py_sq_concat = 40;
constexpr int Py_sq_contains = 41;
constexpr int Py_sq_inplace_concat = 42;
constexpr int Py_sq_inplace_repeat = 43;
constexpr int Py_sq_item = 44;
constexpr int Py_sq_length = 45;
constexpr int Py_sq_repeat = 46;
constexpr int Py_tp_alloc = 47;
constexpr int Py_tp_base = 48;
constexpr int Py_tp_bases = 49;
constexpr int Py_tp_call = 50;
constexpr int Py_tp_clear = 51;
constexpr int Py_tp_dealloc = 52;
constexpr int Py_tp_del = 53;
constexpr int Py_tp_descr_get = 54;
constexpr int Py_tp_descr_set = 55;
constexpr int Py_tp_doc = 56;
constexpr int Py_tp_getattr = 57;
constexpr int Py_tp_getattro = 58;
constexpr int Py_tp_hash = 59;
constexpr int Py_tp_init = 60;
constexpr int Py_tp_is_gc = 61;
constexpr int Py_tp_iter = 62;
constexpr int Py_tp_iternext = 63;
constexpr int Py_tp_methods = 64;
constexpr int Py_tp_new = 65;
constexpr int Py_tp_repr = 66;
constexpr int Py_tp_richcompare = 67;
constexpr int Py_tp_setattr = 68;
constexpr int Py_tp_setattro = 69;
constexpr int Py_tp_str = 70;
constexpr int Py_tp_traverse = 71;
constexpr int Py_tp_members = 72;
constexpr int Py_tp_getset = 73;
constexpr int Py_tp_free = 74;

struct PyType_Slot {
    int slot;    /* slot id, see below */
    void *pfunc; /* function pointer */
};

struct PyType_Spec {
    const char* name;
    int basicsize;
    int itemsize;
    unsigned int flags;
    PyType_Slot *slots; /* terminated by slot==0. */
};

PyObject* PyType_FromSpec(PyType_Spec*);

namespace detail {
// `Py_TPFLAGS_CHECKTYPES` tells Python to pass operator arguments as-is to our operator
// functions. Otherwise, `nb_coerce` must be implemented to coerce values to an instance
// of the given class. `Py_TPFLAGS_CHECKTYPES` provides the same operator interface as
// Python 3.
constexpr int autoclass_base_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES;
}  // namespace detail
#else
namespace detail {
constexpr int autoclass_base_flags = Py_TPFLAGS_DEFAULT;
}  // namespace detail
#endif
}  // namespace py
