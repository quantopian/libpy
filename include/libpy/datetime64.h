#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <ostream>
#include <ratio>
#include <type_traits>

namespace py {
namespace chrono {
using ns = std::chrono::duration<std::int64_t, std::nano>;
using us = std::chrono::duration<std::int64_t, std::micro>;
using ms = std::chrono::duration<std::int64_t, std::milli>;
using s = std::chrono::duration<std::int64_t>;
using m = std::chrono::duration<std::int64_t, std::ratio<60>>;
using h = std::chrono::duration<std::int64_t, std::ratio<60 * 60>>;
using D = std::chrono::duration<std::int64_t, std::ratio<60 * 60 * 24>>;
}  // namespace chrono

/** A datetime64 represented as nanoseconds since 1970-01-01.

    ## Notes
    The comparison operators are "nat-aware" meaning comparisons to
    `datetime64::nat()` always return false.
 */
template<typename Unit>
class datetime64 {
private:
    static constexpr std::int64_t nat_value = std::numeric_limits<std::int64_t>::min();

    template<typename F, typename Rep, typename Period>
    Unit ns_arithmetic(F&& op, const std::chrono::duration<Rep, Period>& d) const {
        if (isnat()) {
            return Unit(nat_value);
        }

        return op(m_value, d);
    }

protected:
    template<typename Unit2>
    friend class datetime64;

    Unit m_value;

public:
    using unit = Unit;

    /** The largest representable datetime64.
     */
    constexpr static datetime64 max() {
        return datetime64(std::numeric_limits<std::int64_t>::max());
    }

    /** The smallest representable datetime64.
     */
    constexpr static datetime64 min() {
        if (std::is_same_v<Unit, chrono::ns>) {
            // The same value as pandas.Timestamp.min.asm8.view('i8'). Currently numpy
            // will overflow an integer in the repr of very negative datetime64s so this
            // is a safe value
            return datetime64(-9223285636854775000LL);
        }

        // min is `nat`, so the smallest valid value is int min + 1
        return datetime64(std::numeric_limits<std::int64_t>::min() + 1);
    }

    /** The special 'not-a-time' value which has `nan` like behavior with
        datetime64 objects.
     */
    constexpr static datetime64 nat() {
        return datetime64(nat_value);
    }

    /** Retrieve the epoch in the given unit.
     */
    constexpr static datetime64 epoch() {
        return datetime64(0);
    }

    /** 1990-01-02, the first day of the zipline NYSE calendar.
     */
    constexpr static datetime64 nyse_epoch() {
        return datetime64(std::chrono::nanoseconds(631238400000000000l));
    }

    /** Default constructor provides a nat value.
     */
    constexpr datetime64() : m_value(nat_value) {}

    /** Unit coercion constructor.
     */
    template<typename Unit2>
    constexpr datetime64(const datetime64<Unit2>& cpfrom) : m_value(cpfrom.m_value) {}

    /** Constructor from the number of `unit` ticks since the epoch as an integral value.

        @param value The number of `unit` ticks since the epoch.
    */
    constexpr explicit datetime64(std::int64_t value) noexcept : m_value(value) {}

    /** Constructor from an offset from the epoch.

        @param value The offset from the epoch.
    */
    template<typename Storage, typename Unit2>
    constexpr explicit datetime64(
        const std::chrono::duration<Storage, Unit2>& value) noexcept
        : m_value(value) {}

    /** Check if a datetime is datetime64::nat()

        @return is this nat?
     */
    constexpr bool isnat() const {
        return m_value.count() == nat_value;
    }

    /** Override for `static_cast<std::int64_t>(datetime64)
     */
    explicit constexpr operator std::int64_t() const {
        return m_value.count();
    }

    /** Compares the datetimes with
        `datetime64::nat().identical(datetime64::nat()) -> true`.

        @param other The datetime to compare to.
        @return Are these the same datetime values?
     */
    constexpr bool identical(const datetime64& other) const {
        return m_value == other.m_value;
    }

    constexpr bool operator==(const datetime64& other) const {
        return m_value == other.m_value && !isnat() && !other.isnat();
    }

    constexpr bool operator!=(const datetime64& other) const {
        return m_value != other.m_value && !isnat() && !other.isnat();
    }

    constexpr bool operator<(const datetime64& other) const {
        return m_value < other.m_value && !isnat() && !other.isnat();
    }

    constexpr bool operator<=(const datetime64& other) const {
        return m_value <= other.m_value && !isnat() && !other.isnat();
    }

    constexpr bool operator>(const datetime64& other) const {
        return m_value > other.m_value && !isnat() && !other.isnat();
    }

    constexpr bool operator>=(const datetime64& other) const {
        return m_value >= other.m_value && !isnat() && !other.isnat();
    }

    template<typename T>
    constexpr datetime64 operator+(const T& d) const {
        return datetime64(ns_arithmetic([](auto a, auto b) { return a + b; }, d));
    }

    template<typename T>
    datetime64& operator+=(const T& d) {
        m_value = ns_arithmetic([](auto a, auto b) { return a + b; }, d);
        return *this;
    }

    template<typename T>
    constexpr datetime64 operator-(const T& d) const {
        return datetime64(ns_arithmetic([](auto a, auto b) { return a - b; }, d));
    }

    template<typename D>
    constexpr datetime64 operator-(const datetime64<D>& d) const {
        if (d.isnat()) {
            return nat();
        }
        return datetime64(ns_arithmetic([](auto a, auto b) { return a - b; }, d.m_value));
    }

    template<typename T>
    datetime64& operator-=(const T& d) {
        m_value = ns_arithmetic([](auto a, auto b) { return a - b; }, d);
        return *this;
    }

    template<typename D>
    datetime64& operator-=(const datetime64<D>& d) {
        if (d.isnat()) {
            return *this = nat();
        }

        m_value = ns_arithmetic([](auto a, auto b) { return a - b; }, d.m_value);
        return *this;
    }
};

using datetime64ns = datetime64<chrono::ns>;
static_assert(std::is_standard_layout<datetime64ns>::value,
              "datetime64 should be standard layout");
static_assert(sizeof(datetime64ns) == sizeof(std::int64_t),
              "alias type should be the same size as aliased type");

template<typename unit>
std::ostream& operator<<(std::ostream& stream, const datetime64<unit>& dt) {
    return stream << static_cast<std::int64_t>(dt);
}
}  // namespace py

namespace std {
template<typename unit>
struct hash<py::datetime64<unit>> {
    using result_type = std::size_t;

    result_type operator()(const py::datetime64<unit>& dt) const noexcept {
        return std::hash<std::int64_t>{}(static_cast<std::int64_t>(dt));
    }
};
}  // namespace std
