#pragma once

#include "libpy/detail/python.h"

namespace py {
/** A wrapper around the threadstate.
 */
struct gil final {
private:
    static thread_local PyThreadState* m_save;

public:
    gil() = delete;

    /** Release the GIL if it is not already released.
     */
    static inline void release() {
        if (held()) {
            m_save = PyEval_SaveThread();
        }
    }

    /** Acquire the GIL if we do not already hold it.
     */
    static inline void acquire() {
        if (!held()) {
            PyEval_RestoreThread(m_save);
            m_save = nullptr;
        }
    }

    /** Check if the GIL is currently held by this thread.

        @return Is the GIL held by this thread?
     */
    static inline bool held() {
        return PyGILState_Check();
    }

    /** RAII resource for ensuring that the gil is released in a given block.

        For example: `auto released = py::gil::release_block{};`
     */
    struct release_block final {
    private:
        bool m_acquire;

    public:
        inline release_block() : m_acquire(gil::held()) {
            gil::release();
        }

        inline void dismiss() {
            m_acquire = false;
            gil::acquire();
        }

        inline ~release_block() {
            if (m_acquire) {
                gil::acquire();
            }
        }
    };

    /** RAII resource for ensuring that the gil is held in a given block.

        For example: `auto held = py::gil::hold_block{};`
     */
    struct hold_block final {
    private:
        bool m_release;

    public:
        inline hold_block() : m_release(!gil::held()) {
            gil::acquire();
        }

        inline void dismiss() {
            m_release = false;
            gil::release();
        }

        inline ~hold_block() {
            if (m_release) {
                gil::release();
            }
        }
    };
};
}  // namespace py
