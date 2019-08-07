#pragma once

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <ratio>
#include <system_error>
#include <tuple>
#include <type_traits>

#include "libpy/char_sequence.h"

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

    template<typename A, typename B = Unit>
    using promote_unit = std::chrono::duration<
        std::int64_t,
        std::conditional_t<
            std::ratio_less_equal_v<typename A::period, typename B::period>,
            typename A::period,
            typename B::period>>;

    template<typename F, typename Other, typename Promotion = promote_unit<Other>>
    constexpr Promotion ns_arithmetic(F&& op, const Other& d) const {
        if (isnat()) {
            return Promotion(nat_value);
        }

        return op(std::chrono::duration_cast<Promotion>(m_value),
                  std::chrono::duration_cast<Promotion>(d));
    }

protected:
    template<typename Unit2>
    friend class datetime64;

    Unit m_value;

public:
    using unit = Unit;

    /** The largest representable datetime64.
     */
    constexpr static datetime64 max() noexcept {
        return datetime64(std::numeric_limits<std::int64_t>::max());
    }

    /** The smallest representable datetime64.
     */
    constexpr static datetime64 min() noexcept {
        if (std::is_same_v<Unit, chrono::ns>) {
            // The same value as pandas.Timestamp.min.asm8.view('i8').
            // Currently numpy will overflow an integer in the repr of very
            // negative datetime64s so this is a safe value
            return datetime64(-9223285636854775000LL);
        }

        // min is `nat`, so the smallest valid value is int min + 1
        return datetime64(std::numeric_limits<std::int64_t>::min() + 1);
    }

    /** The special 'not-a-time' value which has `nan` like behavior with
        datetime64 objects.
     */
    constexpr static datetime64 nat() noexcept {
        return datetime64(nat_value);
    }

    /** Retrieve the epoch in the given unit.
     */
    constexpr static datetime64 epoch() noexcept {
        return datetime64(0);
    }

    /** 1990-01-02, the first day of the zipline NYSE calendar.
     */
    constexpr static datetime64 nyse_epoch() noexcept {
        return datetime64(std::chrono::nanoseconds(631238400000000000l));
    }

    /** Default constructor provides a nat value.
     */
    constexpr datetime64() noexcept : m_value(nat_value) {}

    /** Unit coercion constructor.
     */
    template<typename Unit2>
    constexpr datetime64(const datetime64<Unit2>& cpfrom) noexcept
        : m_value(std::chrono::floor<Unit>(cpfrom.m_value)) {}

    /** Constructor from the number of `unit` ticks since the epoch as an
       integral value.

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
    template<typename OtherUnit>
    constexpr bool identical(const datetime64<OtherUnit>& other) const {
        return m_value == other.m_value;
    }

    constexpr bool operator==(const datetime64& other) const {
        return m_value == other.m_value && !isnat() && !other.isnat();
    }

    constexpr bool operator!=(const datetime64& other) const {
        return isnat() || other.isnat() || m_value != other.m_value;
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
    constexpr auto operator+(const T& d) const {
        auto value = ns_arithmetic([](auto a, auto b) { return a + b; }, d);
        return datetime64<decltype(value)>(value);
    }

    template<typename T>
    datetime64& operator+=(const T& d) {
        m_value = ns_arithmetic([](auto a, auto b) { return a + b; }, d);
        return *this;
    }

    template<typename T>
    constexpr auto operator-(const T& d) const {
        auto value = ns_arithmetic([](auto a, auto b) { return a - b; }, d);
        return datetime64<decltype(value)>(value);
    }

    template<typename D>
    constexpr auto operator-(const datetime64<D>& d) const {
        using out_type = decltype(*this - d.m_value);
        if (d.isnat()) {
            return out_type::nat();
        }
        return *this - d.m_value;
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
        return *this -= d.m_value;
    }
};

using datetime64ns = datetime64<chrono::ns>;
static_assert(std::is_standard_layout<datetime64ns>::value,
              "datetime64 should be standard layout");
static_assert(sizeof(datetime64ns) == sizeof(std::int64_t),
              "alias type should be the same size as aliased type");

namespace chrono {
/* The number of days in each month. The array at index 0 holds the counts for non-leap
   years. The array at index 1 holds the counts for leap years.
*/
static constexpr std::array<std::array<std::int8_t, 12>, 2> days_in_month = {
    // The number of days in each month for non-leap years.
    std::array<std::int8_t, 12>{31,   // jan
                                28,   // feb
                                31,   // mar
                                30,   // apr
                                31,   // may
                                30,   // jun
                                31,   // jul
                                31,   // aug
                                30,   // sep
                                31,   // oct
                                30,   // nov
                                31},  // dec
    // The number of days in each month for leap years.
    std::array<std::int8_t, 12>{31,   // jan
                                29,   // feb
                                31,   // mar
                                30,   // apr
                                31,   // may
                                30,   // jun
                                31,   // jul
                                31,   // aug
                                30,   // sep
                                31,   // oct
                                30,   // nov
                                31},  // dec
};

/** Check if `year` is a leap year.
 */
inline constexpr bool is_leapyear(std::int64_t year) {
    return (year % 4) == 0 && ((year % 100) != 0 || (year % 400) == 0);
}

namespace detail {
inline constexpr std::array<std::array<uint16_t, 12>, 2> build_days_before_month() {
    std::array<std::array<uint16_t, 12>, 2> out = {{0}};
    for (std::size_t n = 1; n < 12; ++n) {
        for (int is_leapyear = 0; is_leapyear < 2; ++is_leapyear) {
            out[is_leapyear][n] = days_in_month[is_leapyear][n - 1] +
                                  out[is_leapyear][n - 1];
        }
    }
    return out;
}

/** The number of days that occur before the first of the month in a non-leap
    year. The array at index 0 holds the counts for non-leap years. The array at
    index 1 holds the counts for leap years.
 */
constexpr std::array<std::array<uint16_t, 12>, 2> days_before_month =
    build_days_before_month();

inline constexpr int leap_years_before(int year) {
    --year;
    return (year / 4) - (year / 100) + (year / 400);
}
}  // namespace detail

/** Compute the time since the unix epoch for a given year,
    month and day.

    @param year
    @month The month, 1-indexed (1 = January)
    @day The day, 1-indexed (1 = The first of the month).
    @return The time since the epoch as a `std::chrono::duration`.
 */
inline constexpr auto time_since_epoch(int year, int month, int day) {
    using days = std::chrono::duration<std::int64_t, std::ratio<86400>>;
    // The number of seconds in 365 days. This doesn't account for leap years, we will
    // manually add those days.
    using years = std::chrono::duration<std::int64_t, std::ratio<31536000>>;

    days out = years(year - 1970);
    out += days(detail::leap_years_before(year) - detail::leap_years_before(1970));
    out += days(detail::days_before_month[is_leapyear(year)][month - 1]);
    out += days(day - 1);

    return out;
}
}  // namespace chrono

namespace detail {
namespace {
using namespace py::cs::literals;
// Constant time access to zero padded numbers suitable for use in the months, days,
// hours, minutes, and seconds field of a datetime.
static constexpr std::array<std::array<char, 2>, 60> datetime_strings =
    {"00"_arr, "01"_arr, "02"_arr, "03"_arr, "04"_arr, "05"_arr, "06"_arr, "07"_arr,
     "08"_arr, "09"_arr, "10"_arr, "11"_arr, "12"_arr, "13"_arr, "14"_arr, "15"_arr,
     "16"_arr, "17"_arr, "18"_arr, "19"_arr, "20"_arr, "21"_arr, "22"_arr, "23"_arr,
     "24"_arr, "25"_arr, "26"_arr, "27"_arr, "28"_arr, "29"_arr, "30"_arr, "31"_arr,
     "32"_arr, "33"_arr, "34"_arr, "35"_arr, "36"_arr, "37"_arr, "38"_arr, "39"_arr,
     "40"_arr, "41"_arr, "42"_arr, "43"_arr, "44"_arr, "45"_arr, "46"_arr, "47"_arr,
     "48"_arr, "49"_arr, "50"_arr, "51"_arr, "52"_arr, "53"_arr, "54"_arr, "55"_arr,
     "56"_arr, "57"_arr, "58"_arr, "59"_arr};

constexpr auto nat_string = "NaT"_arr;

// The maximum size for a datetime string of the given units.
template<typename unit>
constexpr std::int64_t max_size = 0;

template<>
constexpr std::int64_t
    max_size<py::chrono::ns> = "-2262-04-11T23:47:16.854775807"_arr.size();

template<>
constexpr std::int64_t
    max_size<py::chrono::us> = "-294247-01-10T04:00:54.775807"_arr.size();

template<>
constexpr std::int64_t
    max_size<py::chrono::ms> = "-292278994-08-17T07:12:55.807"_arr.size();

template<>
constexpr std::int64_t
    max_size<py::chrono::s> = "-292277026596-12-04T15:30:07"_arr.size();

template<>
constexpr std::int64_t max_size<py::chrono::m> = "-17536621479585-08-30T18:07"_arr.size();

template<>
constexpr std::int64_t max_size<py::chrono::h> = "-1052197288658909-10-10T07"_arr.size();

template<>
constexpr std::int64_t max_size<py::chrono::D> = "-25252734927768524-07-27"_arr.size();
}  // namespace

/** Convert a count of days from 1970 to a year and the number of days into the year

    @param days_from_epoch The number of days since 1970-01-01.
    @return The year number and the number of days into that year.

    @note this is adapted from numpy's `days_to_yearsdays` function
*/
inline constexpr std::tuple<std::int64_t, std::int16_t>
days_to_year_and_days(std::int64_t days_from_epoch) {
    constexpr std::int64_t days_per_400_years = (400 * 365 + 100 - 4 + 1);
    // adjust so it's relative to the year 2000 (divisible by 400)
    std::int64_t days = days_from_epoch - (365 * 30 + 7);
    std::int64_t year = 0;

    // break down the 400 year cycle to get the year and day within the year
    if (days >= 0) {
        year = 400 * (days / days_per_400_years);
        days = days % days_per_400_years;
    }
    else {
        year = 400 * ((days - (days_per_400_years - 1)) / days_per_400_years);
        days = days % days_per_400_years;
        if (days < 0) {
            days += days_per_400_years;
        }
    }

    // work out the year/day within the 400 year cycle
    if (days >= 366) {
        year += 100 * ((days - 1) / (100 * 365 + 25 - 1));
        days = (days - 1) % (100 * 365 + 25 - 1);
        if (days >= 365) {
            year += 4 * ((days + 1) / (4 * 365 + 1));
            days = (days + 1) % (4 * 365 + 1);
            if (days >= 366) {
                year += (days - 1) / 365;
                days = (days - 1) % 365;
            }
        }
    }

    return {year + 2000, days};
}

/** Convert a year and number of days into the year into the month number and day number,
    both 1-indexed.

    @param year The gregorian year.
    @param days_into_year The number of days into the year.
    @return The month number and day number, both 1-indexed.
*/
inline constexpr std::tuple<std::int8_t, std::int8_t>
month_day_for_year_days(std::int64_t year, std::int16_t days_into_year) {
    const auto& month_lengths = chrono::days_in_month[chrono::is_leapyear(year)];

    for (int ix = 0; ix < 12; ++ix) {
        if (days_into_year < month_lengths[ix]) {
            return {ix + 1, days_into_year + 1};
        }
        else {
            days_into_year -= month_lengths[ix];
        }
    }
    __builtin_unreachable();
}

namespace formatting {
inline void write(char* data, char*, int& ix, char v) {
    data[ix++] = v;
}

inline void write(char* data, char*, int& ix, char c, std::int64_t count) {
    std::memset(data + ix, c, count);
    ix += count;
}

inline void write(char* data, char* end, int& ix, std::int64_t v) {
    auto begin = data + ix;
    auto [p, ec] = std::to_chars(data + ix, end, v);
    if (ec != std::errc()) {
        throw std::runtime_error("failed to format integer");
    }
    ix += p - begin;
}

template<std::size_t size>
void write(char* data, char*, int& ix, const std::array<char, size>& v) {
    std::memcpy(data + ix, v.data(), size);
    ix += size;
}

inline void write(char* data, char*, int& ix, const std::string_view& v) {
    std::memcpy(data + ix, v.data(), v.size());
    ix += v.size();
}
}  // namespace formatting
}  // namespace detail

template<typename unit>
std::to_chars_result
to_chars(char* first, char* last, const datetime64<unit>& dt, bool compress = false) {
    if (last - first < detail::max_size<unit>) {
        return {first, std::errc::value_too_large};
    }

    if (dt.isnat()) {
        // format all `NaT` values in a uniform way, regardless of the unit
        std::memcpy(first, detail::nat_string.data(), detail::nat_string.size());
        return {first + detail::nat_string.size(), std::errc()};
    }

    int ix = 0;
    auto write = [&](auto... args) {
        detail::formatting::write(first, last, ix, args...);
    };

    auto zero_pad = [&](int expected_digits, std::int64_t value) {
        std::int64_t digits = std::floor(std::log10(value));
        digits += 1;
        if (expected_digits > digits) {
            write('0', expected_digits - digits);
        }
    };

    auto finalize = [&]() -> std::to_chars_result { return {first + ix, std::errc()}; };

    datetime64<chrono::D> as_days(dt);
    auto [year, days_into_year] = detail::days_to_year_and_days(
        static_cast<std::int64_t>(as_days));
    auto [month, day] = detail::month_day_for_year_days(year, days_into_year);

    if (year < 0 && year > -100) {
        write('-');
        year = std::abs(year);
        zero_pad(3, year);
    }
    else if (year > 0) {
        zero_pad(4, year);
    }
    write(year);
    write('-');
    write(detail::datetime_strings[month]);
    write('-');
    write(detail::datetime_strings[day]);
    if (std::is_same_v<unit, py::chrono::D> || (compress && dt == as_days)) {
        return finalize();
    }

    write('T');
    datetime64<chrono::h> as_hours(dt);
    write(detail::datetime_strings[std::abs(
        static_cast<std::int64_t>(as_hours - as_days))]);
    if (std::is_same_v<unit, py::chrono::h>) {
        return finalize();
    }

    write(':');
    datetime64<chrono::m> as_minutes(dt);
    write(detail::datetime_strings[static_cast<std::int64_t>(as_minutes - as_hours)]);
    if (std::is_same_v<unit, py::chrono::m>) {
        return finalize();
    }

    write(':');
    datetime64<chrono::s> as_seconds(dt);
    write(detail::datetime_strings[static_cast<std::int64_t>(as_seconds - as_minutes)]);
    if (std::is_same_v<unit, py::chrono::s> || (compress && dt == as_seconds)) {
        return finalize();
    }

    auto fractional_seconds = static_cast<std::int64_t>(dt - as_seconds);
    if (fractional_seconds) {
        write('.');
        std::int64_t expected_digits = std::log10(unit::period::den);
        zero_pad(expected_digits, fractional_seconds);
        write(fractional_seconds);
    }
    return finalize();
}

template<typename unit>
std::ostream& operator<<(std::ostream& stream, const datetime64<unit>& dt) {
    std::array<char, detail::max_size<unit>> data;
    auto [end, errc] = to_chars(data.begin(), data.end(), dt);
    if (errc != std::errc()) {
        throw std::runtime_error("failed to format datetime64");
    }
    return stream.write(data.begin(), end - data.begin());
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
