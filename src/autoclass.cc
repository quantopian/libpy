#include <cstring>
#include <forward_list>
#include <typeindex>
#include <unordered_map>

#include <Python.h>
#include <object.h>
#include <structmember.h>

#include "libpy/detail/autoclass_cache.h"
#include "libpy/detail/autoclass_py2.h"

namespace py::detail {
std::unordered_map<std::type_index, autoclass_storage> autoclass_type_cache{};
}  // namespace py::detail

// The following section is taken mostly from CPython as a backport of the
// `PyType_FromSpec` API. Minor modifications are made to support Python 2, like
// converting PyUnicode* calls to PyString*
#if PY_MAJOR_VERSION == 2
namespace py {
namespace detail {
namespace {
constexpr short slotoffsets[] = {
    -1, /* invalid slot */
    0,
    0,
    offsetof(PyHeapTypeObject, as_mapping.mp_ass_subscript),
    offsetof(PyHeapTypeObject, as_mapping.mp_length),
    offsetof(PyHeapTypeObject, as_mapping.mp_subscript),
    offsetof(PyHeapTypeObject, as_number.nb_absolute),
    offsetof(PyHeapTypeObject, as_number.nb_add),
    offsetof(PyHeapTypeObject, as_number.nb_and),
    offsetof(PyHeapTypeObject, as_number.nb_nonzero),
    offsetof(PyHeapTypeObject, as_number.nb_divmod),
    offsetof(PyHeapTypeObject, as_number.nb_float),
    offsetof(PyHeapTypeObject, as_number.nb_floor_divide),
    offsetof(PyHeapTypeObject, as_number.nb_index),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_add),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_and),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_floor_divide),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_lshift),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_multiply),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_or),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_power),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_remainder),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_rshift),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_subtract),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_true_divide),
    offsetof(PyHeapTypeObject, as_number.nb_inplace_xor),
    offsetof(PyHeapTypeObject, as_number.nb_int),
    offsetof(PyHeapTypeObject, as_number.nb_invert),
    offsetof(PyHeapTypeObject, as_number.nb_lshift),
    offsetof(PyHeapTypeObject, as_number.nb_multiply),
    offsetof(PyHeapTypeObject, as_number.nb_negative),
    offsetof(PyHeapTypeObject, as_number.nb_or),
    offsetof(PyHeapTypeObject, as_number.nb_positive),
    offsetof(PyHeapTypeObject, as_number.nb_power),
    offsetof(PyHeapTypeObject, as_number.nb_remainder),
    offsetof(PyHeapTypeObject, as_number.nb_rshift),
    offsetof(PyHeapTypeObject, as_number.nb_subtract),
    offsetof(PyHeapTypeObject, as_number.nb_true_divide),
    offsetof(PyHeapTypeObject, as_number.nb_xor),
    offsetof(PyHeapTypeObject, as_sequence.sq_ass_item),
    offsetof(PyHeapTypeObject, as_sequence.sq_concat),
    offsetof(PyHeapTypeObject, as_sequence.sq_contains),
    offsetof(PyHeapTypeObject, as_sequence.sq_inplace_concat),
    offsetof(PyHeapTypeObject, as_sequence.sq_inplace_repeat),
    offsetof(PyHeapTypeObject, as_sequence.sq_item),
    offsetof(PyHeapTypeObject, as_sequence.sq_length),
    offsetof(PyHeapTypeObject, as_sequence.sq_repeat),
    offsetof(PyHeapTypeObject, ht_type.tp_alloc),
    offsetof(PyHeapTypeObject, ht_type.tp_base),
    offsetof(PyHeapTypeObject, ht_type.tp_bases),
    offsetof(PyHeapTypeObject, ht_type.tp_call),
    offsetof(PyHeapTypeObject, ht_type.tp_clear),
    offsetof(PyHeapTypeObject, ht_type.tp_dealloc),
    offsetof(PyHeapTypeObject, ht_type.tp_del),
    offsetof(PyHeapTypeObject, ht_type.tp_descr_get),
    offsetof(PyHeapTypeObject, ht_type.tp_descr_set),
    offsetof(PyHeapTypeObject, ht_type.tp_doc),
    offsetof(PyHeapTypeObject, ht_type.tp_getattr),
    offsetof(PyHeapTypeObject, ht_type.tp_getattro),
    offsetof(PyHeapTypeObject, ht_type.tp_hash),
    offsetof(PyHeapTypeObject, ht_type.tp_init),
    offsetof(PyHeapTypeObject, ht_type.tp_is_gc),
    offsetof(PyHeapTypeObject, ht_type.tp_iter),
    offsetof(PyHeapTypeObject, ht_type.tp_iternext),
    offsetof(PyHeapTypeObject, ht_type.tp_methods),
    offsetof(PyHeapTypeObject, ht_type.tp_new),
    offsetof(PyHeapTypeObject, ht_type.tp_repr),
    offsetof(PyHeapTypeObject, ht_type.tp_richcompare),
    offsetof(PyHeapTypeObject, ht_type.tp_setattr),
    offsetof(PyHeapTypeObject, ht_type.tp_setattro),
    offsetof(PyHeapTypeObject, ht_type.tp_str),
    offsetof(PyHeapTypeObject, ht_type.tp_traverse),
    offsetof(PyHeapTypeObject, ht_type.tp_members),
    offsetof(PyHeapTypeObject, ht_type.tp_getset),
    offsetof(PyHeapTypeObject, ht_type.tp_free),
    // offsetof(PyHeapTypeObject, as_number.nb_matrix_multiply),
    // offsetof(PyHeapTypeObject, as_number.nb_inplace_matrix_multiply),
    // offsetof(PyHeapTypeObject, as_async.am_await),
    // offsetof(PyHeapTypeObject, as_async.am_aiter),
    // offsetof(PyHeapTypeObject, as_async.am_anext),
    // offsetof(PyHeapTypeObject, ht_type.tp_finalize),
};

#define PyTrash_UNWIND_LEVEL 50

#define Py_TRASHCAN_BEGIN_CONDITION(op, cond)                                            \
    do {                                                                                 \
        PyThreadState* _tstate = NULL;                                                   \
        /* If "cond" is false, then _tstate remains NULL and the deallocator             \
         * is run normally without involving the trashcan */                             \
        if (cond) {                                                                      \
            _tstate = PyThreadState_GET();                                               \
            if (_tstate->trash_delete_nesting >= PyTrash_UNWIND_LEVEL) {                 \
                /* Store the object (to be deallocated later) and jump past              \
                 * Py_TRASHCAN_END, skipping the body of the deallocator */              \
                _PyTrash_thread_deposit_object((PyObject*) (op));                        \
                break;                                                                   \
            }                                                                            \
            ++_tstate->trash_delete_nesting;                                             \
        }
/* The body of the deallocator is here. */
#define Py_TRASHCAN_END                                                                  \
    if (_tstate) {                                                                       \
        --_tstate->trash_delete_nesting;                                                 \
        if (_tstate->trash_delete_later && _tstate->trash_delete_nesting <= 0)           \
            _PyTrash_thread_destroy_chain();                                             \
    }                                                                                    \
    }                                                                                    \
    while (0)                                                                            \
        ;

#define Py_TRASHCAN_BEGIN(op, dealloc)                                                   \
    Py_TRASHCAN_BEGIN_CONDITION(op, Py_TYPE(op)->tp_dealloc == (destructor)(dealloc))

static void clear_slots(PyTypeObject* type, PyObject* self) {
    Py_ssize_t i, n;
    PyMemberDef* mp;

    n = Py_SIZE(type);
    mp = PyHeapType_GET_MEMBERS((PyHeapTypeObject*) type);
    for (i = 0; i < n; i++, mp++) {
        if (mp->type == T_OBJECT_EX && !(mp->flags & READONLY)) {
            char* addr = (char*) self + mp->offset;
            PyObject* obj = *(PyObject**) addr;
            if (obj != NULL) {
                *(PyObject**) addr = NULL;
                Py_DECREF(obj);
            }
        }
    }
}

static void subtype_dealloc(PyObject* self) {
    PyTypeObject *type, *base;
    destructor basedealloc;
    int has_finalizer;

    /* Extract the type; we expect it to be a heap type */
    type = Py_TYPE(self);

    /* Test whether the type has GC exactly once */

    if (!PyType_IS_GC(type)) {
        /* It's really rare to find a dynamic type that doesn't have
           GC; it can only happen when deriving from 'object' and not
           adding any slots or instance variables.  This allows
           certain simplifications: there's no need to call
           clear_slots(), or DECREF the dict, or clear weakrefs. */
        if (type->tp_del) {
            type->tp_del(self);
            if (self->ob_refcnt > 0)
                return;
        }

        /* Find the nearest base with a different tp_dealloc */
        base = type;
        while ((basedealloc = base->tp_dealloc) == subtype_dealloc) {
            assert(Py_SIZE(base) == 0);
            base = base->tp_base;
            assert(base);
        }

        /* Extract the type again; tp_del may have changed it */
        type = Py_TYPE(self);

        /* Call the base tp_dealloc() */
        assert(basedealloc);
        basedealloc(self);

        /* Can't reference self beyond this point */
        Py_DECREF(type);

        /* Done */
        return;
    }

    /* We get here only if the type has GC */

    /* UnTrack and re-Track around the trashcan macro, alas */
    /* See explanation at end of function for full disclosure */
    PyObject_GC_UnTrack(self);
    Py_TRASHCAN_BEGIN(self, subtype_dealloc);

    /* Find the nearest base with a different tp_dealloc */
    base = type;
    while ((/*basedealloc =*/base->tp_dealloc) == subtype_dealloc) {
        base = base->tp_base;
        assert(base);
    }

    has_finalizer = (bool) type->tp_del;
    /*
      If we added a weaklist, we clear it. Do this *before* calling tp_del,
      clearing slots, or clearing the instance dict.

      GC tracking must be off at this point. weakref callbacks (if any, and
      whether directly here or indirectly in something we call) may trigger GC,
      and if self is tracked at that point, it will look like trash to GC and GC
      will try to delete self again.
    */
    if (type->tp_weaklistoffset && !base->tp_weaklistoffset)
        PyObject_ClearWeakRefs(self);

    if (type->tp_del) {
        _PyObject_GC_TRACK(self);
        type->tp_del(self);
        if (self->ob_refcnt > 0) {
            /* Resurrected */
            goto endlabel;
        }
        _PyObject_GC_UNTRACK(self);
    }
    if (has_finalizer) {
        /* New weakrefs could be created during the finalizer call.
           If this occurs, clear them out without calling their
           finalizers since they might rely on part of the object
           being finalized that has already been destroyed. */
        if (type->tp_weaklistoffset && !base->tp_weaklistoffset) {
            /* Modeled after GET_WEAKREFS_LISTPTR() */
            PyWeakReference** list = (PyWeakReference**) PyObject_GET_WEAKREFS_LISTPTR(
                self);
            while (*list)
                _PyWeakref_ClearRef(*list);
        }
    }

    /*  Clear slots up to the nearest base with a different tp_dealloc */
    base = type;
    while ((basedealloc = base->tp_dealloc) == subtype_dealloc) {
        if (Py_SIZE(base))
            clear_slots(base, self);
        base = base->tp_base;
        assert(base);
    }

    /* If we added a dict, DECREF it */
    if (type->tp_dictoffset && !base->tp_dictoffset) {
        PyObject** dictptr = _PyObject_GetDictPtr(self);
        if (dictptr != NULL) {
            PyObject* dict = *dictptr;
            if (dict != NULL) {
                Py_DECREF(dict);
                *dictptr = NULL;
            }
        }
    }

    /* Extract the type again; tp_del may have changed it */
    type = Py_TYPE(self);

    /* Call the base tp_dealloc(); first retrack self if
     * basedealloc knows about gc.
     */
    if (PyType_IS_GC(base))
        _PyObject_GC_TRACK(self);
    assert(basedealloc);
    basedealloc(self);

    /* Can't reference self beyond this point. It's possible tp_del switched
       our type from a HEAPTYPE to a non-HEAPTYPE, so be careful about
       reference counting. */
    if (type->tp_flags & Py_TPFLAGS_HEAPTYPE)
        Py_DECREF(type);

endlabel:
    Py_TRASHCAN_END

    /* Explanation of the weirdness around the trashcan macros:

       Q. What do the trashcan macros do?

       A. Read the comment titled "Trashcan mechanism" in CPython's object.h.
          For one, this explains why there must be a call to GC-untrack
          before the trashcan begin macro.      Without understanding the
          trashcan code, the answers to the following questions don't make
          sense.

       Q. Why do we GC-untrack before the trashcan and then immediately
          GC-track again afterward?

       A. In the case that the base class is GC-aware, the base class
          probably GC-untracks the object.      If it does that using the
          UNTRACK macro, this will crash when the object is already
          untracked.  Because we don't know what the base class does, the
          only safe thing is to make sure the object is tracked when we
          call the base class dealloc.  But...  The trashcan begin macro
          requires that the object is *untracked* before it is called.  So
          the dance becomes:

         GC untrack
         trashcan begin
         GC track

       Q. Why did the last question say "immediately GC-track again"?
          It's nowhere near immediately.

       A. Because the code *used* to re-track immediately.      Bad Idea.
          self has a refcount of 0, and if gc ever gets its hands on it
          (which can happen if any weakref callback gets invoked), it
          looks like trash to gc too, and gc also tries to delete self
          then.  But we're already deleting self.  Double deallocation is
          a subtle disaster.
    */
}

static int extra_ivars(PyTypeObject* type, PyTypeObject* base) {
    size_t t_size = type->tp_basicsize;
    size_t b_size = base->tp_basicsize;

    assert(t_size >= b_size); /* Else type smaller than base! */
    if (type->tp_itemsize || base->tp_itemsize) {
        /* If itemsize is involved, stricter rules */
        return t_size != b_size || type->tp_itemsize != base->tp_itemsize;
    }
    if (type->tp_weaklistoffset && base->tp_weaklistoffset == 0 &&
        type->tp_weaklistoffset + sizeof(PyObject*) == t_size &&
        type->tp_flags & Py_TPFLAGS_HEAPTYPE)
        t_size -= sizeof(PyObject*);
    if (type->tp_dictoffset && base->tp_dictoffset == 0 &&
        type->tp_dictoffset + sizeof(PyObject*) == t_size &&
        type->tp_flags & Py_TPFLAGS_HEAPTYPE)
        t_size -= sizeof(PyObject*);

    return t_size != b_size;
}

static PyTypeObject* solid_base(PyTypeObject* type) {
    PyTypeObject* base;

    if (type->tp_base)
        base = solid_base(type->tp_base);
    else
        base = &PyBaseObject_Type;
    if (extra_ivars(type, base))
        return type;
    else
        return base;
}

static PyTypeObject* best_base(PyObject* bases) {
    Py_ssize_t i, n;
    PyTypeObject *base, *winner, *candidate, *base_i;
    PyObject* base_proto;

    assert(PyTuple_Check(bases));
    n = PyTuple_GET_SIZE(bases);
    assert(n > 0);
    base = NULL;
    winner = NULL;
    for (i = 0; i < n; i++) {
        base_proto = PyTuple_GET_ITEM(bases, i);
        if (!PyType_Check(base_proto)) {
            PyErr_SetString(PyExc_TypeError, "bases must be types");
            return NULL;
        }
        base_i = (PyTypeObject*) base_proto;
        if (base_i->tp_dict == NULL) {
            if (PyType_Ready(base_i) < 0)
                return NULL;
        }
        if (!PyType_HasFeature(base_i, Py_TPFLAGS_BASETYPE)) {
            PyErr_Format(PyExc_TypeError,
                         "type '%.100s' is not an acceptable base type",
                         base_i->tp_name);
            return NULL;
        }
        candidate = solid_base(base_i);
        if (winner == NULL) {
            winner = candidate;
            base = base_i;
        }
        else if (PyType_IsSubtype(winner, candidate))
            ;
        else if (PyType_IsSubtype(candidate, winner)) {
            winner = candidate;
            base = base_i;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "multiple bases have "
                            "instance lay-out conflict");
            return NULL;
        }
    }
    assert(base != NULL);

    return base;
}

#define Py_ARRAY_LENGTH(array) (sizeof(array) / sizeof((array)[0]))

PyObject* PyType_FromSpecWithBases(PyType_Spec* spec, PyObject* bases) {
    PyHeapTypeObject* res;
    PyMemberDef* memb;
    PyObject* modname;
    PyTypeObject *type, *base;

    PyType_Slot* slot;
    Py_ssize_t nmembers;
    const char *s, *res_start;

    nmembers = 0;
    for (slot = spec->slots; slot->slot; slot++) {
        if (slot->slot == Py_tp_members) {
            nmembers = 0;
            for (memb = (PyMemberDef*) (slot->pfunc); memb->name != NULL; memb++) {
                nmembers++;
            }
        }
    }

    res = (PyHeapTypeObject*) PyType_GenericAlloc(&PyType_Type, nmembers);
    if (res == NULL)
        return NULL;
    res_start = (char*) res;

    if (spec->name == NULL) {
        PyErr_SetString(PyExc_SystemError, "Type spec does not define the name field.");
        goto fail;
    }

    /* Set the type name and qualname */
    s = strrchr(spec->name, '.');
    if (s == NULL)
        s = (char*) spec->name;
    else
        s++;

    type = &res->ht_type;
    /* The flags must be initialized early, before the GC traverses us */
    type->tp_flags = spec->flags | Py_TPFLAGS_HEAPTYPE;
    res->ht_name = PyString_FromString(s);
    if (!res->ht_name)
        goto fail;
    type->tp_name = spec->name;

    /* Adjust for empty tuple bases */
    if (!bases) {
        base = &PyBaseObject_Type;
        /* See whether Py_tp_base(s) was specified */
        for (slot = spec->slots; slot->slot; slot++) {
            if (slot->slot == Py_tp_base)
                base = (PyTypeObject*) slot->pfunc;
            else if (slot->slot == Py_tp_bases) {
                bases = (PyObject*) slot->pfunc;
                Py_INCREF(bases);
            }
        }
        if (!bases)
            bases = PyTuple_Pack(1, base);
        if (!bases)
            goto fail;
    }
    else
        Py_INCREF(bases);

    /* Calculate best base, and check that all bases are type objects */
    base = best_base(bases);
    if (base == NULL) {
        goto fail;
    }
    if (!PyType_HasFeature(base, Py_TPFLAGS_BASETYPE)) {
        PyErr_Format(PyExc_TypeError,
                     "type '%.100s' is not an acceptable base type",
                     base->tp_name);
        goto fail;
    }

    /* Initialize essential fields */
    type->tp_as_number = &res->as_number;
    type->tp_as_sequence = &res->as_sequence;
    type->tp_as_mapping = &res->as_mapping;
    type->tp_as_buffer = &res->as_buffer;
    /* Set tp_base and tp_bases */
    type->tp_bases = bases;
    bases = NULL;
    Py_INCREF(base);
    type->tp_base = base;

    type->tp_basicsize = spec->basicsize;
    type->tp_itemsize = spec->itemsize;

    for (slot = spec->slots; slot->slot; slot++) {
        if (slot->slot < 0 || (size_t) slot->slot >= Py_ARRAY_LENGTH(slotoffsets)) {
            PyErr_SetString(PyExc_RuntimeError, "invalid slot offset");
            goto fail;
        }
        else if (slot->slot == Py_tp_base || slot->slot == Py_tp_bases) {
            /* Processed above */
            continue;
        }
        else if (slot->slot == Py_tp_doc) {
            /* For the docstring slot, which usually points to a static string
               literal, we need to make a copy */
            const char* old_doc = (const char*) slot->pfunc;
            size_t len = strlen(old_doc) + 1;
            char* tp_doc = (char*) PyObject_MALLOC(len);
            if (tp_doc == NULL) {
                type->tp_doc = NULL;
                PyErr_NoMemory();
                goto fail;
            }
            memcpy(tp_doc, old_doc, len);
            type->tp_doc = tp_doc;
        }
        else if (slot->slot == Py_tp_members) {
            /* Move the slots to the heap type itself */
            size_t len = Py_TYPE(type)->tp_itemsize * nmembers;
            memcpy(PyHeapType_GET_MEMBERS(res), slot->pfunc, len);
            type->tp_members = PyHeapType_GET_MEMBERS(res);
        }
        else {
            /* Copy other slots directly */
            *(void**) (res_start + slotoffsets[slot->slot]) = slot->pfunc;
        }
    }
    if (type->tp_dealloc == NULL) {
        /* It's a heap type, so needs the heap types' dealloc.
           subtype_dealloc will call the base type's tp_dealloc, if
           necessary. */
        type->tp_dealloc = subtype_dealloc;
    }

    if (PyType_Ready(type) < 0)
        goto fail;

    /* Set type.__module__ */
    s = strrchr(spec->name, '.');
    if (s != NULL) {
        int err;
        modname = PyString_FromStringAndSize(spec->name, (Py_ssize_t)(s - spec->name));
        if (modname == NULL) {
            goto fail;
        }
        err = PyDict_SetItemString(type->tp_dict, "__module__", modname);
        Py_DECREF(modname);
        if (err != 0)
            goto fail;
    }

    return (PyObject*) res;

fail:
    Py_DECREF(res);
    return NULL;
}
}  // namespace
}  // namespace detail

PyObject* PyType_FromSpec(PyType_Spec* spec) {
    return detail::PyType_FromSpecWithBases(spec, nullptr);
}
// End section taken from CPython
}  // namespace py
#endif
