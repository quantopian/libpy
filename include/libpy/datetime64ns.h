#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>

namespace py {
/** A datetime64 represented as nanoseconds since 1970-01-01.

    ## Notes
    The comparison operators are "nat-aware" meaning comparisons to
    `datetime64::nat()` always return false.
 */
class datetime64ns {
private:
    static constexpr std::int64_t nat_value = std::numeric_limits<std::int64_t>::min();

    std::int64_t m_value;

    /** Helper for adapting a time point to ns since the epoch.

     */
    template<typename Tp>
    static inline auto to_ns(const Tp& tp) noexcept {
        auto as_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(tp);
        return as_ns.time_since_epoch().count();
    }

    template<typename F, typename Rep, typename Period>
    inline std::int64_t
    ns_arithmetic(F&& op, const std::chrono::duration<Rep, Period>& duration) const {
        if (isnat()) {
            return nat_value;
        }

        auto count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        return op(m_value, count);
    }

public:
    /** The largest representable datetime64ns.
     */
    constexpr inline static datetime64ns max() {
        return datetime64ns(std::numeric_limits<std::int64_t>::max());
    }

    /** The smallest representable datetime64ns.
     */
    constexpr inline static datetime64ns min() {
        return datetime64ns(-9223285636854775000l);
    }

    /** The special 'not-a-time' value which has `nan` like behavior with
        datetime64ns objects.
     */
    constexpr inline static datetime64ns nat() {
        return datetime64ns(nat_value);
    }

    /** 1990-01-02, the first day of the zipline NYSE calendar.
     */
    constexpr inline static datetime64ns nyse_epoch() {
        return datetime64ns(631238400000000000l);
    }

    /** Default constructor provides a nat value.
     */
    constexpr inline datetime64ns() : m_value(nat_value) {}

    /** Copy constructor.
     */
    constexpr inline datetime64ns(const datetime64ns& cpfrom) : m_value(cpfrom.m_value) {}

    /** Constructor from the number of ns since the epoch as an integral value.

        @param value The number of nanoseconds since the epoch.
    */
    constexpr inline explicit datetime64ns(std::int64_t value) noexcept
        : m_value(value) {}

    /** Constructor from `std::chrono::time_point`

        @param tp The time point to adapt.
    */
    template<typename Clock, typename Duration>
    constexpr inline datetime64ns(
        const std::chrono::time_point<Clock, Duration>& tp) noexcept
        : m_value(to_ns(tp)) {}

    /** Check if a datetime is datetime64ns::nat()

        @return is this nat?
     */
    constexpr inline bool isnat() const {
        return m_value == nat_value;
    }

    /** Override for `static_cast<std::int64_t>(datetime64ns)
     */
    explicit constexpr operator std::int64_t() const {
        return m_value;
    }

    /** Compares the datetimes with
        `datetime64ns::nat().identical(datetime64ns::nat()) -> true`.

        @param other The datetime to compare to.
        @return Are these the same datetime values?
     */
    constexpr inline bool identical(const datetime64ns& other) const {
        return m_value == other.m_value;
    }

    constexpr inline bool operator==(const datetime64ns& other) const {
        return m_value == other.m_value && !isnat() && !other.isnat();
    }

    constexpr inline bool operator!=(const datetime64ns& other) const {
        return m_value != other.m_value && !isnat() && !other.isnat();
    }

    constexpr inline bool operator<(const datetime64ns& other) const {
        return m_value < other.m_value && !isnat() && !other.isnat();
    }

    constexpr inline bool operator<=(const datetime64ns& other) const {
        return m_value <= other.m_value && !isnat() && !other.isnat();
    }

    constexpr inline bool operator>(const datetime64ns& other) const {
        return m_value > other.m_value && !isnat() && !other.isnat();
    }

    constexpr inline bool operator>=(const datetime64ns& other) const {
        return m_value >= other.m_value && !isnat() && !other.isnat();
    }

    template<typename T>
    constexpr inline datetime64ns operator+(const T& duration) const {
        return datetime64ns(
            ns_arithmetic([](auto a, auto b) { return a + b; }, duration));
    }

    template<typename T>
    datetime64ns& operator+=(const T& duration) {
        m_value = ns_arithmetic([](auto a, auto b) { return a + b; }, duration);

        return *this;
    }

    template<typename T>
    constexpr inline datetime64ns operator-(const T& duration) const {
        return datetime64ns(
            ns_arithmetic([](auto a, auto b) { return a - b; }, duration));
    }

    template<typename T>
    datetime64ns& operator-=(const T& duration) {
        m_value = ns_arithmetic([](auto a, auto b) { return a - b; }, duration);

        return *this;
    }
};
static_assert(std::is_standard_layout<datetime64ns>::value,
              "datetime64ns should be standard layout");
static_assert(sizeof(datetime64ns) == sizeof(std::int64_t),
              "alias type should be the same size as aliased type");

std::ostream& operator<<(std::ostream& stream, const datetime64ns& dt) {
    return stream << static_cast<std::int64_t>(dt);
}
}  // namespace py
