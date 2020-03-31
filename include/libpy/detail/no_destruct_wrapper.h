#pragma once

#include <memory>

namespace py::detail {
/** A wrapper which prevents a destructor from being called. This is useful for
    static objects that hold `py::owned_ref` objects which may not be able
    to be cleaned up do to the interpreter state.
 */
template<typename T>
class no_destruct_wrapper {
public:
    /** Forward all arguments to the underlying object.
     */
    template<typename... Args>
    no_destruct_wrapper(Args... args) {
        new (&m_storage) T(std::forward<Args>(args)...);
    }

    T& get() {
        return *std::launder(reinterpret_cast<T*>(&m_storage));
    }

    const T& get() const {
        return *std::launder(reinterpret_cast<const T*>(&m_storage));
    }

private:
    std::aligned_storage_t<sizeof(T), alignof(T)> m_storage;
};
}  // namespace libpy::detail
