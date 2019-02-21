#pragma once

#include <any>

namespace py {
/** A unique function for dispatching to the `operator=` of another type.

    @tparam The type to dispatch to.
    @param lhs_addr The address of the left hand side of the assignment.
    @param rhs_addr The address of the right hand side of the assignment.
 */
template<typename T>
void any_ref_assign(void* lhs_addr, const void* rhs_addr) {
    *static_cast<T*>(lhs_addr) = *static_cast<const T*>(rhs_addr);
}

/** A function pointer to a specialization of `any_ref_assign`. This doubles
    as the run time type information tag for any ref and view.
 */
using any_ref_assign_func = void (*)(void*, const void*);

/** A mutable dynamic reference to a type which isn't known until runtime.

    This object is like `std::any`, except it is *non-owning*. Assignment has
    reference semantics, meaning it will assign through to the referent.

    @see make_any_ref
 */
class any_ref {
private:
    void* m_addr;
    any_ref_assign_func m_assign;

public:
    /** Construct an `any_ref` from some arbitrary address and an assignment
        function. The assignment function pointer doubles as the run time type
        tag.

        @param addr The address of the referent.
        @param assign The function to use to assign to `addr`. This should
                      be a specialization of `any_ref_assign`.
     */
    inline any_ref(void* addr, any_ref_assign_func assign)
        : m_addr(addr), m_assign(assign) {}

    inline any_ref& operator=(const any_ref& rhs) {
        if (rhs.m_assign != m_assign) {
            throw std::bad_any_cast{};
        }

        m_assign(m_addr, rhs.m_addr);
        return *this;
    }

    template<typename T>
    inline any_ref& operator=(const T& rhs) {
        if (&any_ref_assign<T> != m_assign) {
            throw std::bad_any_cast{};
        }

        m_assign(m_addr, std::addressof(rhs));
        return *this;
    }

    /** Cast the dynamic reference to a statically typed reference. If the
        referred object is not of type `T`, and `std::bad_any_cast` will be
        thrown.
     */
    template<typename T>
    T& cast() {
        if (&any_ref_assign<T> != m_assign) {
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
        if (&any_ref_assign<T> != m_assign) {
            throw std::bad_any_cast{};
        }

        return *static_cast<const T*>(m_addr);
    }
};

/** Convert a statically typed reference into a dynamically typed reference.

    @param The reference to convert.
    @return The dynamically typed reference.
 */
template<typename T>
any_ref make_any_ref(T& ob) {
    return {&ob, &any_ref_assign<T>};
}

/** A constant dynamic reference to a type which isn't known until runtime.

    This object is like `std::any`, except it is *non-owning*. The object is
    like a constant reference, and thus may not be assigned to.

    @see make_any_cref
 */
class any_cref {
private:
    const void* m_addr;
    any_ref_assign_func m_assign;

public:
    /** Construct an `any_cref` from some arbitrary address and an assignment
        function. The assignment function pointer doubles as the run time type
        tag, but will never be called (because the reference is immutable).

        @param addr The address of the referent.
        @param assign The function to use to assign to `addr`. This should
                      be a specialization of `any_ref_assign`.
     */
    inline any_cref(const void* addr, any_ref_assign_func assign)
        : m_addr(addr), m_assign(assign) {}

    // movable but not assignable
    any_cref(any_cref&&) = default;
    any_cref& operator=(any_cref&&) = default;
    any_cref(const any_cref&) = delete;
    any_cref& operator=(const any_cref&) = delete;

    /** Cast the dynamic reference to a statically typed reference. If the
        referred object is not of type `T`, and `std::bad_any_cast` will be
        thrown.
     */
    template<typename T>
    const T& cast() const {
        if (&any_ref_assign<T> != m_assign) {
            throw std::bad_any_cast{};
        }

        return *static_cast<const T*>(m_addr);
    }
};

/** Convert a statically typed reference into a dynamically typed reference.

    @param The reference to convert.
    @return The dynamically typed reference.
 */
template<typename T>
any_cref make_any_cref(T& ob) {
    return {&ob, &any_ref_assign<T>};
}
}  // namespace py
