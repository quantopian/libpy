#pragma once
#include <optional>
#include <utility>

namespace py::util {
/** Assign an arbitrary callback to run when the scope closes, either through an exception
    or return. This callback may be dismissed later. This is useful for implementing
    transactional behavior, where all operations either succeed or fail together.

    ### Example

    \code
    // add objects to all vectors, if an exception is thrown, no objects will be added
    //  to any vectors.
    void add_objects(std::vector<A>& as,
                     const A& a,
                     std::vector<B>& bs,
                     const B& b,
                     std::vector<C>& cs,
                     const C& c) {
        as.push_back(a);
        py::util::scope_guard a_guard([&] { as.pop_back(); });

        bs.push_back(b);
        py::util::scope_guard b_guard([&] { bs.pop_back(); });

        cs.push_back(c);

        // everything that could fail has already run, if we make it here we succeeded so
        // we can dismiss the guards
        a_guard.dismiss();
        b_guard.dismiss();
    }
    \endcode
*/
template<typename F>
struct scope_guard {
private:
    std::optional<F> m_callback;

public:
    scope_guard(F&& callback) : m_callback(std::move(callback)) {}

    /** Dismiss the scope guard causing the registered callback to not be called.
     */
    void dismiss() {
        m_callback = std::nullopt;
    }

    ~scope_guard() {
        if (m_callback) {
            (*m_callback)();
        }
    }
};
}  // namespace py::util
