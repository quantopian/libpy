#pragma once

#include <any>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <typeinfo>

#include "libpy/demangle.h"
#include "libpy/exception.h"
#include "libpy/meta.h"
#include "libpy/numpy_utils.h"
#include "libpy/to_object.h"

namespace py {
namespace detail {
struct any_vtable_impl {
    const std::type_info& type_info;
    std::size_t size;
    std::size_t align;
    bool move_is_noexcept;
    bool is_trivially_destructible;
    bool is_trivially_default_constructible;
    bool is_trivially_move_constructible;
    bool is_trivially_copy_constructible;
    bool is_trivially_copyable;
    void (*copy_assign)(void* lhs, const void* rhs);
    void (*move_assign)(void* lhs, void* rhs);
    void (*default_construct)(void* dest);
    void (*copy_construct)(void* dest, const void* value);
    void (*move_construct)(void* dest, void* value);
    void (*destruct)(void* addr);
    bool (*ne)(const void* lhs, const void* rhs);
    bool (*eq)(const void* lhs, const void* rhs);
    py::owned_ref<> (*to_object)(const void* addr);
    py::owned_ref<PyArray_Descr> (*new_dtype)();
    std::ostream& (*ostream_format)(std::ostream& stream, const void* addr);
    std::string (*type_name)();
};

template<typename T>
struct has_ostream_format {
private:
    template<typename U>
    static decltype((std::declval<std::ostream&>() << std::declval<U>()),
                    std::true_type{})
    test(int);

    template<typename>
    static std::false_type test(long);

public:
    static constexpr bool value = std::is_same_v<decltype(test<T>(0)), std::true_type>;
};

/** The actual structure that holds all the function pointers. This should be accessed
    through an `py::any_vtable`.

    NOTE: Do not explicitly specialize this unless you really know what you are doing.
 */
template<typename T>
inline constexpr any_vtable_impl any_vtable_instance = {
    typeid(T),
    sizeof(T),
    alignof(T),
    std::is_nothrow_move_constructible_v<T>,
    std::is_trivially_destructible_v<T>,
    std::is_trivially_default_constructible_v<T>,
    std::is_trivially_move_constructible_v<T>,
    std::is_trivially_copy_constructible_v<T>,
    std::is_trivially_copyable_v<T>,
    [](void* lhs, const void* rhs) {
        *static_cast<T*>(lhs) = *static_cast<const T*>(rhs);
    },
    [](void* lhs, void* rhs) {
        *static_cast<T*>(lhs) = std::move(*static_cast<T*>(rhs));
    },
    [](void* dest) { new (dest) T(); },
    [](void* dest, const void* value) { new (dest) T(*static_cast<const T*>(value)); },
    [](void* dest, void* value) { new (dest) T(std::move(*static_cast<T*>(value))); },
    [](void* addr) { static_cast<T*>(addr)->~T(); },
    [](const void* lhs, const void* rhs) -> bool {
        return *static_cast<const T*>(lhs) != *static_cast<const T*>(rhs);
    },
    [](const void* lhs, const void* rhs) -> bool {
        return *static_cast<const T*>(lhs) == *static_cast<const T*>(rhs);
    },
    []([[maybe_unused]] const void* addr) -> owned_ref<> {
        if constexpr (py::has_to_object<T>) {
            return py::to_object(*static_cast<const T*>(addr));
        }
        else {
            throw py::exception(PyExc_TypeError,
                                "cannot convert values of type ",
                                py::util::type_name<T>(),
                                " into Python object");
        }
    },
    []() -> owned_ref<PyArray_Descr> {
        if constexpr (py::has_new_dtype<T>) {
            return py::new_dtype<T>();
        }
        else {
            throw py::exception(PyExc_TypeError,
                                "cannot create a dtype from the vtable for type:",
                                py::util::type_name<T>());
        }
    },
    []([[maybe_unused]] std::ostream& stream,
       [[maybe_unused]] const void* addr) -> std::ostream& {
        if constexpr (has_ostream_format<const T&>::value) {
            return stream << *static_cast<const T*>(addr);
        }
        else {
            throw py::exception(
                PyExc_TypeError,
                "cannot use operator<<(std::ostream&, const T&) for values of type ",
                py::util::type_name<T>());
        }
    },
    []() { return py::util::type_name<T>(); },
};

[[noreturn]] inline void void_vtable() {
    throw std::runtime_error("cannot use void vtable");
}

template<>
inline constexpr any_vtable_impl any_vtable_instance<void> = {
    typeid(void),
    0,
    0,
    false,
    false,
    false,
    false,
    false,
    false,
    [](void*, const void*) { void_vtable(); },
    [](void*, void*) { void_vtable(); },
    [](void*) { void_vtable(); },
    [](void*, const void*) { void_vtable(); },
    [](void*, void*) { void_vtable(); },
    [](void*) { void_vtable(); },
    [](const void*, const void*) -> bool { void_vtable(); },
    [](const void*, const void*) -> bool { void_vtable(); },
    [](const void*) -> owned_ref<> { void_vtable(); },
    []() -> owned_ref<PyArray_Descr> { void_vtable(); },
    [](std::ostream&, const void*) -> std::ostream& { void_vtable(); },
    []() { return py::util::type_name<void>(); },
};
}  // namespace detail

/** A collection of operations and metadata about a type to implement type-erased
    containers.

    The constructor is private, use `py::any_vtable::make<T>()` to look up the vtable for
    a given type.
 */
class any_vtable {
private:
    const detail::any_vtable_impl* m_impl;

    inline constexpr any_vtable(const detail::any_vtable_impl* impl) : m_impl(impl) {}

public:
    constexpr any_vtable() : any_vtable(&detail::any_vtable_instance<void>) {}

    template<typename T>
    static inline constexpr any_vtable make() {
        return &detail::any_vtable_instance<py::meta::remove_cvref<T>>;
    }

    /** Get access to the underlying collection of function pointers.
     */
    constexpr inline const detail::any_vtable_impl* impl() const {
        return m_impl;
    }

    constexpr inline const std::type_info& type_info() const {
        return m_impl->type_info;
    }

    constexpr inline std::size_t size() const {
        return m_impl->size;
    }

    constexpr inline std::size_t align() const {
        return m_impl->align;
    }

    constexpr inline bool is_trivially_default_constructible() const {
        return m_impl->is_trivially_default_constructible;
    }

    constexpr inline bool is_trivially_destructible() const {
        return m_impl->is_trivially_destructible;
    }

    constexpr inline bool is_trivially_move_constructible() const {
        return m_impl->is_trivially_move_constructible;
    }

    constexpr inline bool is_trivially_copy_constructible() const {
        return m_impl->is_trivially_copy_constructible;
    }

    constexpr inline bool is_trivially_copyable() const {
        return m_impl->is_trivially_copyable;
    }

    inline void copy_assign(void* lhs, const void* rhs) const {
        return m_impl->copy_assign(lhs, rhs);
    }

    inline void move_assign(void* lhs, void* rhs) const {
        return m_impl->copy_assign(lhs, rhs);
    }

    inline void default_construct(void* dest) const {
        return m_impl->default_construct(dest);
    }

    inline void copy_construct(void* dest, const void* value) const {
        return m_impl->copy_construct(dest, value);
    }

    inline void move_construct(void* dest, void* value) const {
        return m_impl->move_construct(dest, value);
    }

    inline void destruct(void* dest) const {
        return m_impl->destruct(dest);
    }

    inline bool move_is_noexcept() const {
        return m_impl->move_is_noexcept;
    }

    inline void move_if_noexcept(void* dest, void* value) const {
        if (m_impl->move_is_noexcept) {
            m_impl->move_construct(dest, value);
        }
        else {
            m_impl->copy_construct(dest, value);
        }
    }

    inline bool ne(const void* lhs, const void* rhs) const {
        return m_impl->ne(lhs, rhs);
    }

    inline bool eq(const void* lhs, const void* rhs) const {
        return m_impl->eq(lhs, rhs);
    }

    inline owned_ref<> to_object(const void* addr) const {
        return m_impl->to_object(addr);
    }

    inline owned_ref<PyArray_Descr> new_dtype() const {
        return m_impl->new_dtype();
    }

    inline std::ostream& ostream_format(std::ostream& stream, const void* addr) const {
        return m_impl->ostream_format(stream, addr);
    }

    inline std::string type_name() const {
        return m_impl->type_name();
    }

    /** Allocate uninitialized memory for `count` objects of the given type.
     */
    inline std::byte* alloc(std::size_t count) const {
        return new (std::align_val_t{align()}) std::byte[size() * count];
    }

    /** Allocate memory and default construct `count` objects of the given type.
     */
    inline std::byte* default_construct_alloc(std::size_t count) const {
        if (is_trivially_default_constructible()) {
            return new (std::align_val_t{align()}) std::byte[size() * count]();
        }
        std::byte* out = alloc(count);
        std::byte* data = out;
        std::byte* end = out + size() * count;
        try {
            for (; data < end; data += size()) {
                default_construct(data);
            }
        }
        catch (...) {
            for (std::byte* p = out; p < data; p += size()) {
                destruct(p);
            }
            free(out);
            throw;
        }

        return out;
    }

    /** Free memory allocated with `alloc` or `default_construct_alloc`.
     */
    inline void free(std::byte* addr) const {
        operator delete[](addr, std::align_val_t{align()});
    }

    /** Check if this vtable refers to the same type as another vtable.
     */
    constexpr inline bool operator==(const any_vtable& other) const {
        // The m_impl can be the same based on optimization/linking; however, it is not
        // guaranteed to be the same if the type is the same. If `m_impl` is the same, we
        // know it must point to the same type, but if not, fall back to the slower
        // `type_info::operator=`.
        return m_impl == other.m_impl || m_impl->type_info == other.m_impl->type_info;
    }

    /** Check if this vtable does not refer to the same type as another vtable.
     */
    constexpr inline bool operator!=(const any_vtable& other) const {
        return !(*this == other);
    }
};

class any_cref;

/** A mutable dynamic reference to a value whose type isn't known until runtime.

    This object is like `std::any`, except it is *non-owning*. Assignment has
    reference semantics, meaning it will assign through to the referent.

    @see make_any_ref
 */
class any_ref {
private:
    void* m_addr;
    any_vtable m_vtable;

    template<typename T>
    void typecheck(const T&) const {
        if (any_vtable::make<T>() != m_vtable) {
            throw std::bad_any_cast{};
        }
    }

    inline void typecheck(const any_ref& other) const {
        if (m_vtable != other.m_vtable) {
            throw std::bad_any_cast{};
        }
    }

    inline void typecheck(const any_cref& other) const;

    template<typename T>
    bool cmp(bool (*f)(const void*, const void*), const T& other) const {
        typecheck(other);
        if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
            return f(m_addr, other.addr());
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
    inline any_ref(void* addr, const any_vtable& vtable)
        : m_addr(addr), m_vtable(vtable) {}

    inline any_ref(const any_ref& cpfrom)
        : m_addr(cpfrom.m_addr), m_vtable(cpfrom.m_vtable) {}

    inline any_ref& operator=(const any_ref& rhs) {
        typecheck(rhs);
        m_vtable.copy_assign(m_addr, rhs.m_addr);
        return *this;
    }

    inline any_ref& operator=(const any_cref& rhs);

    template<typename T>
    any_ref& operator=(const T& rhs) {
        typecheck(rhs);
        m_vtable.copy_assign(m_addr, std::addressof(rhs));
        return *this;
    }

    template<typename T>
    bool operator!=(const T& rhs) const {
        return cmp(m_vtable.impl()->ne, rhs);
    }

    template<typename T>
    bool operator==(const T& rhs) const {
        return cmp(m_vtable.impl()->eq, rhs);
    }

    const any_vtable& vtable() const {
        return m_vtable;
    }

    /** Cast the dynamic reference to a statically typed reference. If the
        referred object is not of type `T`, and `std::bad_any_cast` will be
        thrown.
     */
    template<typename T>
    T& cast() {
        if (any_vtable::make<T>() != m_vtable) {
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
        if (any_vtable::make<T>() != m_vtable) {
            throw std::bad_any_cast{};
        }

        return *static_cast<const T*>(m_addr);
    }

    /** Get the address of the referred-to object.
     */
    inline void* addr() {
        return m_addr;
    }

    /** Get the address of the referred-to object.
     */
    inline const void* addr() const {
        return m_addr;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const any_ref& value) {
    return value.vtable().ostream_format(stream, value.addr());
}

/** Convert a statically typed reference into a dynamically typed reference.

    @param The reference to convert.
    @return The dynamically typed reference.
 */
template<typename T>
any_ref make_any_ref(T& ob) {
    return {std::addressof(ob), any_vtable::make<T>()};
}

/** A constant dynamic reference to a value whose type isn't known until runtime.

    This object is like `std::any`, except it is *non-owning*. The object is
    like a constant reference, and thus may not be assigned to.

    @see make_any_cref
 */
class any_cref {
private:
    const void* m_addr;
    any_vtable m_vtable;

    template<typename T>
    void typecheck(const T&) const {
        if (any_vtable::make<T>() != m_vtable) {
            throw std::bad_any_cast{};
        }
    }

    inline void typecheck(const any_ref& other) const {
        if (m_vtable != other.vtable()) {
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
        if constexpr (std::is_same_v<T, any_cref> || std::is_same_v<T, any_ref>) {
            return f(m_addr, other.addr());
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
    inline any_cref(const void* addr, const any_vtable& vtable)
        : m_addr(addr), m_vtable(vtable) {}

    // movable but not assignable
    any_cref(any_cref&&) = default;
    any_cref& operator=(any_cref&&) = default;
    any_cref(const any_cref&) = delete;
    any_cref& operator=(const any_cref&) = delete;

    template<typename T>
    bool operator!=(const T& rhs) const {
        return cmp(m_vtable.impl()->ne, rhs);
    }

    template<typename T>
    bool operator==(const T& rhs) const {
        return cmp(m_vtable.impl()->eq, rhs);
    }

    inline const any_vtable& vtable() const {
        return m_vtable;
    }

    /** Cast the dynamic reference to a statically typed reference. If the
        referred object is not of type `T`, and `std::bad_any_cast` will be
        thrown.
     */
    template<typename T>
    const T& cast() const {
        if (any_vtable::make<T>() != m_vtable) {
            throw std::bad_any_cast{};
        }

        return *static_cast<const T*>(m_addr);
    }

    /** Get the address of the referred-to object.
     */
    inline const void* addr() const {
        return m_addr;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const any_cref& value) {
    return value.vtable().ostream_format(stream, value.addr());
}

/** Convert a statically typed reference into a dynamically typed reference.

    @param The reference to convert.
    @return The dynamically typed reference.
 */
template<typename T>
any_cref make_any_cref(T& ob) {
    return {std::addressof(ob), any_vtable::make<T>()};
}

// Deferred definition because this deeds to see the definition of `any_cref` to call
// methods. This breaks the cycle between these objects.
inline void any_ref::typecheck(const any_cref& other) const {
    if (m_vtable != other.vtable()) {
        throw std::bad_any_cast{};
    }
}

inline any_ref& any_ref::operator=(const any_cref& rhs) {
    typecheck(rhs);
    m_vtable.copy_assign(m_addr, rhs.addr());
    return *this;
}

namespace dispatch {
/** Convert an any_ref into a Python object.
 */
template<>
struct to_object<py::any_ref> {
    static py::owned_ref<> f(const py::any_ref& ref) {
        return ref.vtable().to_object(ref.addr());
    }
};

template<>
struct to_object<py::any_cref> {
    static py::owned_ref<> f(const py::any_cref& ref) {
        return ref.vtable().to_object(ref.addr());
    }
};
}  // namespace dispatch

namespace detail {
template<std::size_t max_size, typename T>
struct make_string_vtable_impl;

template<std::size_t max_size, std::size_t head, std::size_t... tail>
struct make_string_vtable_impl<max_size, std::index_sequence<head, tail...>> {
    static any_vtable f(int size) {
        switch (size) {
        case head:
            return any_vtable::make<std::array<char, head>>();
        default:
            return make_string_vtable_impl<max_size, std::index_sequence<tail...>>::f(
                size);
        }
    }
};

template<std::size_t max_size>
struct make_string_vtable_impl<max_size, std::index_sequence<>> {
    [[noreturn]] static any_vtable f(int) {
        throw py::exception(
            PyExc_TypeError,
            "cannot create vtable for fixed width strings with size greater than ",
            max_size - 1);
    }
};

inline any_vtable make_string_vtable(int size) {
    constexpr std::size_t max_size = 64;
    return make_string_vtable_impl<max_size, std::make_index_sequence<max_size>>::f(size);
}
}  // namespace detail

/** Lookup the proper any_vtable for the given numpy dtype.

    @param dtype The runtime numpy dtype.
    @return The any_vtable that corresponds to the given dtype.
 */
inline any_vtable dtype_to_vtable(py::borrowed_ref<PyArray_Descr> dtype) {
    switch (dtype->type_num) {
    case NPY_BOOL:
        return any_vtable::make<py_bool>();
    case NPY_INT8:
        return any_vtable::make<std::int8_t>();
    case NPY_INT16:
        return any_vtable::make<std::int16_t>();
    case NPY_INT32:
        return any_vtable::make<std::int32_t>();
    case NPY_INT64:
        return any_vtable::make<std::int64_t>();
    case NPY_UINT8:
        return any_vtable::make<std::uint8_t>();
    case NPY_UINT16:
        return any_vtable::make<std::uint16_t>();
    case NPY_UINT32:
        return any_vtable::make<std::uint32_t>();
    case NPY_UINT64:
        return any_vtable::make<std::uint64_t>();
    case NPY_FLOAT32:
        return any_vtable::make<float>();
    case NPY_FLOAT64:
        return any_vtable::make<double>();
    case NPY_DATETIME:
        switch (auto unit = reinterpret_cast<PyArray_DatetimeDTypeMetaData*>(
                                dtype->c_metadata)
                                ->meta.base) {
        case py_chrono_unit_to_numpy_unit<py::chrono::ns>:
            return any_vtable::make<py::datetime64<py::chrono::ns>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::us>:
            return any_vtable::make<py::datetime64<py::chrono::us>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::ms>:
            return any_vtable::make<py::datetime64<py::chrono::ms>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::s>:
            return any_vtable::make<py::datetime64<py::chrono::s>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::m>:
            return any_vtable::make<py::datetime64<py::chrono::m>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::h>:
            return any_vtable::make<py::datetime64<py::chrono::h>>();
        case py_chrono_unit_to_numpy_unit<py::chrono::D>:
            return any_vtable::make<py::datetime64<py::chrono::D>>();
        case NPY_FR_GENERIC:
            throw exception(PyExc_TypeError, "cannot adapt unitless datetime");
        default:
            throw exception(PyExc_TypeError, "unknown datetime unit: ", unit);
        }
    case NPY_OBJECT:
        return any_vtable::make<owned_ref<>>();
    case NPY_STRING:
        return detail::make_string_vtable(dtype->elsize);
    }

    throw py::exception(PyExc_TypeError,
                        "cannot create an any ref view over an ndarray of dtype: ",
                        static_cast<PyObject*>(dtype));
}
}  // namespace py

namespace std {
template<>
struct hash<py::any_vtable> {
    std::size_t operator()(const py::any_vtable& vtable) const {
        return vtable.type_info().hash_code();
    }
};
}  // namespace std
