#pragma once

#include <any>

#include "libpy/demangle.h"

namespace py {
struct any_ref_vtable_type {
    std::size_t size;
    void (*assign)(void* lhs, const void* rhs);
    bool (*ne)(const void* lhs, const void* rhs);
    bool (*eq)(const void* lhs, const void* rhs);
    void (*default_construct)(void* dest);
    void (*copy_construct)(void* dest, const void* value);
    void (*move_construct)(void* dest, void* value);
    py::util::demangled_cstring (*type_name)();
};

namespace detail {
/** A unique function for dispatching to the `operator=` of another type.

    @tparam The type to dispatch to.
    @param lhs_addr The address of the left hand side of the assignment.
    @param rhs_addr The address of the right hand side of the assignment.
 */
template<typename T>
constexpr any_ref_vtable_type any_ref_vtable_instance = {
    sizeof(T),
    [](void* lhs, const void* rhs) {
        *static_cast<T*>(lhs) = *static_cast<const T*>(rhs);
    },
    [](const void* lhs, const void* rhs) -> bool {
        return *static_cast<const T*>(lhs) != *static_cast<const T*>(rhs);
    },
    [](const void* lhs, const void* rhs) -> bool {
        return *static_cast<const T*>(lhs) == *static_cast<const T*>(rhs);
    },
    [](void* dest) {
        new(dest) T();
    },
    [](void* dest, const void* value) {
        new(dest) T(*static_cast<const T*>(value));
    },
    [](void* dest, void* value) {
        new(dest) T(std::move(*static_cast<T*>(value)));
    },
    []() {
        return py::util::type_name<T>();
    },
};
}  // namespace detail

template<typename T>
constexpr const any_ref_vtable_type* any_ref_vtable = &detail::any_ref_vtable_instance<T>;

/** A mutable dynamic reference to a value whose type isn't known until runtime.

    This object is like `std::any`, except it is *non-owning*. Assignment has
    reference semantics, meaning it will assign through to the referent.

    @see make_any_ref
 */
class any_ref {
private:
    void* m_addr;
    const any_ref_vtable_type* m_vtable;

    template<typename T>
    void typecheck(const T&) const {
        if (any_ref_vtable<T> != m_vtable) {
            throw std::bad_any_cast{};
        }
    }

    inline void typecheck(const any_ref& other) const {
        if (m_vtable != other.m_vtable) {
            throw std::bad_any_cast{};
        }
    }

    template<typename T>
    bool cmp(bool (*f)(const void*, const void*), const T& other) const {
        typecheck(other);
        if constexpr (std::is_same_v<T, any_ref>) {
            return f(m_addr, other.m_addr);
        }
        else {
            return f(m_addr, std::addressof(other));
        }
    }

public:
    /** Construct an `any_ref` from some arbitrary address and an assignment
        function. The assignment function pointer doubles as the run time type
        tag.

        @param addr The address of the referent.
        @param vtable The vtable for the type of the referent.
     */
    inline any_ref(void* addr, const any_ref_vtable_type* vtable)
        : m_addr(addr), m_vtable(vtable) {}

    inline any_ref& operator=(const any_ref& rhs) {
        typecheck(rhs);
        m_vtable->assign(m_addr, rhs.m_addr);
        return *this;
    }

    template<typename T>
    any_ref& operator=(const T& rhs) {
        typecheck(rhs);
        m_vtable->assign(m_addr, std::addressof(rhs));
        return *this;
    }

    template<typename T>
    bool operator!=(const T& rhs) const {
        return cmp(m_vtable->ne, rhs);
    }

    template<typename T>
    bool operator==(const T& rhs) const {
        return cmp(m_vtable->eq, rhs);
    }

    const any_ref_vtable_type* vtable() const {
        return m_vtable;
    }

    /** Cast the dynamic reference to a statically typed reference. If the
        referred object is not of type `T`, and `std::bad_any_cast` will be
        thrown.
     */
    template<typename T>
    T& cast() {
        if (any_ref_vtable<T> != m_vtable) {
            throw std::bad_any_cast{};
        }

        return *static_cast<T*>(m_addr);
    }

    /** Cast the dynamic reference to a statically typed reference. If the
        referred object is not of type `T`, and `std::bad_any_cast` will be
        thrown.
     */
    template<typename T>
    const T& cast() const {
        if (any_ref_vtable<T> != m_vtable) {
            throw std::bad_any_cast{};
        }

        return *static_cast<const T*>(m_addr);
    }

    inline void* addr() {
        return m_addr;
    }

    inline const void* addr() const {
        return m_addr;
    }
};

/** Convert a statically typed reference into a dynamically typed reference.

    @param The reference to convert.
    @return The dynamically typed reference.
 */
template<typename T>
any_ref make_any_ref(T& ob) {
    return {&ob, any_ref_vtable<T>};
}

/** A constant dynamic reference to a value whose type isn't known until runtime.

    This object is like `std::any`, except it is *non-owning*. The object is
    like a constant reference, and thus may not be assigned to.

    @see make_any_cref
 */
class any_cref {
private:
    const void* m_addr;
    const any_ref_vtable_type* m_vtable;

    template<typename T>
    void typecheck(const T&) const {
        if (any_ref_vtable<T> != m_vtable) {
            throw std::bad_any_cast{};
        }
    }

    inline void typecheck(const any_cref& other) const {
        if (m_vtable != other.m_vtable) {
            throw std::bad_any_cast{};
        }
    }

    template<typename T>
    bool cmp(bool (*f)(const void*, const void*), const T& other) const {
        typecheck(other);
        if constexpr (std::is_same_v<T, any_cref>) {
            return f(m_addr, other.m_addr);
        }
        else {
            return f(m_addr, std::addressof(other));
        }
    }

public:
    /** Construct an `any_cref` from some arbitrary address and an assignment
        function. The assignment function pointer doubles as the run time type
        tag, but will never be called (because the reference is immutable).

        @param addr The address of the referent.
        @param vtable The vtable for the type of the referent.
     */
    inline any_cref(const void* addr, const any_ref_vtable_type* vtable)
        : m_addr(addr), m_vtable(vtable) {}

    // movable but not assignable
    any_cref(any_cref&&) = default;
    any_cref& operator=(any_cref&&) = default;
    any_cref(const any_cref&) = delete;
    any_cref& operator=(const any_cref&) = delete;

    template<typename T>
    bool operator!=(const T& rhs) const {
        return cmp(m_vtable->ne, rhs);
    }

    template<typename T>
    bool operator==(const T& rhs) const {
        return cmp(m_vtable->eq, rhs);
    }

    const any_ref_vtable_type* vtable() const {
        return m_vtable;
    }

    /** Cast the dynamic reference to a statically typed reference. If the
        referred object is not of type `T`, and `std::bad_any_cast` will be
        thrown.
     */
    template<typename T>
    const T& cast() const {
        if (any_ref_vtable<T> != m_vtable) {
            throw std::bad_any_cast{};
        }

        return *static_cast<const T*>(m_addr);
    }

    inline const void* addr() const {
        return m_addr;
    }
};

/** Convert a statically typed reference into a dynamically typed reference.

    @param The reference to convert.
    @return The dynamically typed reference.
 */
template<typename T>
any_cref make_any_cref(T& ob) {
    return {&ob, any_ref_vtable<T>};
}
}  // namespace py
