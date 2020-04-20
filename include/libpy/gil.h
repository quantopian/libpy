#pragma once

#include "libpy/detail/api.h"
#include "libpy/detail/python.h"

namespace py {
LIBPY_BEGIN_EXPORT
/** A wrapper around the threadstate.
 */
struct gil final {
private:
    LIBPY_EXPORT static thread_local PyThreadState* m_save;

public:
    gil() = delete;

    /** Release the GIL. The GIL must be held.

        @note `release` is a low-level utility. Please see `release_block` for
              a safer alternative.
     */
    static inline void release() {
        m_save = PyEval_SaveThread();
    }

    /** Acquire the GIL. The GIL must not be held.

        @note `acquire` is a low-level utility. Please see `hold_block` for
              a safer alternative.
     */
    static inline void acquire() {
        PyEval_RestoreThread(m_save);
        m_save = nullptr;
    }

    /** Release the GIL if it is not already released.

        @note `ensure_released` is a low-level utility. Please see
              `release_block` for a safer alternative.
     */
    static inline void ensure_released() {
        if (held()) {
            release();
        }
    }

    /** Acquire the GIL if we do not already hold it.

        @note `ensure_acquired` is a low-level utility. Please see `hold_block`
              for a safer alternative.
     */
    static inline void ensure_acquired() {
        if (!held()) {
            acquire();
        }
    }

    static inline bool held() {
        return PyGILState_Check();
    }

    /** RAII resource for ensuring that the gil is released in a given block.

        For example: `py::gil::release_block released;`
     */
    struct release_block final {
    private:
        bool m_acquire;

    public:
        inline release_block() : m_acquire(gil::held()) {
            gil::ensure_released();
        }

        inline void dismiss() {
            m_acquire = false;
            gil::ensure_acquired();
        }

        inline ~release_block() {
            if (m_acquire) {
                gil::acquire();
            }
        }
    };

    /** RAII resource for ensuring that the gil is held in a given block.

        For example: `py::gil::hold_block held;`
     */
    struct hold_block final {
    private:
        bool m_release;

    public:
        inline hold_block() : m_release(!gil::held()) {
            gil::ensure_acquired();
        }

        inline void dismiss() {
            m_release = false;
            gil::ensure_released();
        }

        inline ~hold_block() {
            if (m_release) {
                gil::release();
            }
        }
    };
};
LIBPY_END_EXPORT
}  // namespace py
