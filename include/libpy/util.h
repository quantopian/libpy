#pragma once

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "libpy/borrowed_ref.h"
#include "libpy/detail/python.h"
#include "libpy/owned_ref.h"

/** Miscellaneous utilities.
 */
namespace py::util {
/** Format a string by building up an intermediate `std::stringstream`.

    @param msg The components of the message.
    @return A new string which formatted all parts of msg with
            `operator(std::ostream&, decltype(msg))`
 */
template<typename... Ts>
std::string format_string(Ts&&... msg) {
    std::stringstream s;
    (s << ... << msg);
    return s.str();
}

/** Create an exception object by forwarding the result of `msg` to
    `py::util::format_string`.

    @tparam Exc The type of the exception to raise.
    @param msg The components of the message.
    @return A new exception object.
 */
template<typename Exc, typename... Args>
Exc formatted_error(Args&&... msg) {
    return Exc(format_string(std::forward<Args>(msg)...));
}

/** Check if all parameters are equal.
 */
template<typename T, typename... Ts>
constexpr bool all_equal(T&& head, Ts&&... tail) {
    return (... && (head == tail));
}

constexpr inline bool all_equal() {
    return true;
}

/** Extract a C-style string from a `str` object.

    The result will be a view into a cached utf-8 representation of `ob`.

    The lifetime of the returned value is the same as the lifetime of `ob`.
*/
inline const char* pystring_to_cstring(py::borrowed_ref<> ob) {
    return PyUnicode_AsUTF8(ob.get());
}

/** Get a view over the contents of a `str`.

    The view will be over a cached utf-8 representation of `ob`.

    The lifetime of the returned value is the same as the lifetime of `ob`.
*/
inline std::string_view pystring_to_string_view(py::borrowed_ref<> ob) {
    Py_ssize_t size;
    const char* cs;

    cs = PyUnicode_AsUTF8AndSize(ob.get(), &size);
    if (!cs) {
        throw formatted_error<std::runtime_error>(
            "failed to get string and size from object of type: ",
            Py_TYPE(ob.get())->tp_name);
    }
    return {cs, static_cast<std::size_t>(size)};
}

/* Taken from google benchmark, this is useful for debugging.

   The DoNotOptimize(...) function can be used to prevent a value or
   expression from being optimized away by the compiler. This function is
   intended to add little to no overhead.
   See: https://youtu.be/nXaxk27zwlk?t=2441
*/
template<typename T>
inline __attribute__((always_inline)) void do_not_optimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template<typename T>
inline __attribute__((always_inline)) void do_not_optimize(T& value) {
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

/** Find lower bound index for needle within contianer.
 */
template<typename C, typename T>
std::int64_t searchsorted_l(const C& container, const T& needle) {
    auto begin = container.begin();
    return std::lower_bound(begin, container.end(), needle) - begin;
}

/** Find upper bound index for needle within container.
 */
template<typename C, typename T>
std::int64_t searchsorted_r(const C& container, const T& needle) {
    auto begin = container.begin();
    return std::upper_bound(begin, container.end(), needle) - begin;
}

/** Call `f` with value, start and stop (exclusive) indices for each contiguous region in
    `it` of equal value.
 */
template<typename I, typename F>
void apply_to_groups(I begin, I end, F&& f) {
    if (begin == end) {
        return;
    }

    std::size_t start_ix = 0;
    auto previous = *begin;
    ++begin;

    std::size_t ix = 1;
    for (; begin != end; ++begin, ++ix) {
        auto value = *begin;
        if (value == previous) {
            continue;
        }

        f(previous, start_ix, ix);
        start_ix = ix;
        previous = value;
    }

    f(previous, start_ix, ix);
}

template<typename R, typename F>
void apply_to_groups(R&& range, F&& f) {
    apply_to_groups(range.begin(), range.end(), f);
}
}  // namespace py::util
