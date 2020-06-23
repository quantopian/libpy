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

    /** Check if the gil is currently held.
     */
    static inline bool held() {
        return PyGILState_Check();
    }

    /** RAII resource for ensuring that the gil is released in a given block.

        For example:

        \code
        // the gil may or may not be released here
        {
            py::gil::release_block released;
            // the gil is now definitely released
        }
        // the gil may or may not be released here
        \endcode
     */
    struct release_block final {
    private:
        bool m_acquire;

    public:
        inline release_block() : m_acquire(gil::held()) {
            gil::ensure_released();
        }

        /** Reset this gil back to the state it was in when this object was created.
         */
        inline void dismiss() {
            if (m_acquire) {
                gil::acquire();
                m_acquire = false;
            }
        }

        inline ~release_block() {
            dismiss();
        }
    };

    /** RAII resource for ensuring that the gil is held in a given block.

        For example:

        \code
        // the gil may or may not be held here
        {
            py::gil::hold_block held;
            // the gil is now definitely held
        }
        // the gil may or may not be held here
        \endcode
     */
    struct hold_block final {
    private:
        bool m_release;

    public:
        inline hold_block() : m_release(!gil::held()) {
            gil::ensure_acquired();
        }

        /** Reset this gil back to the state it was in when this object was created.
         */
        inline void dismiss() {
            if (m_release) {
                gil::release();
                m_release = false;
            }
        }

        inline ~hold_block() {
            dismiss();
        }
    };
};
LIBPY_END_EXPORT
}  // namespace py
