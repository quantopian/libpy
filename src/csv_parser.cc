#include <mutex>
#include <numeric>
#include <thread>

#include "libpy/autoclass.h"
#include "libpy/detail/csv_parser.h"
#include "libpy/exception.h"
#include "libpy/getattr.h"
#include "libpy/numpy_utils.h"
#include "libpy/util.h"

#if LIBPY_NO_CSV_PREFETCH
#define LIBPY_CSV_PREFETCH(...)
#else
#define LIBPY_CSV_PREFETCH(...) __builtin_prefetch(__VA_ARGS__)
#endif

namespace py::csv::parser {

IMPORT_ARRAY_MODULE_SCOPE();

void cell_parser::set_num_lines(std::size_t) {}
void cell_parser::set_num_threads(int) {}

namespace {
// Allow us to configure the parser for a different l1dcache line size at compile time.
#ifndef LIBPY_L1DCACHE_LINE_SIZE
#define LIBPY_L1DCACHE_LINE_SIZE 64
#endif

constexpr int l1dcache_line_size = LIBPY_L1DCACHE_LINE_SIZE;

constexpr std::size_t min_split_lines_bytes_size = 16384;

class header_parser : public cell_parser {
protected:
    std::vector<std::string> m_parsed;

public:
    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t,
                                        std::string_view row,
                                        std::size_t offset,
                                        int) override {
        auto& cell = m_parsed.emplace_back();
        return std::get<1>(detail::chomp_quoted_string([&](char c) { cell.push_back(c); },
                                                       delim,
                                                       row,
                                                       offset));
    }

    const std::vector<std::string>& parsed() const {
        return m_parsed;
    }
};
}  // namespace

std::tuple<std::size_t, bool> skip_parser::chomp(char delim,
                                                 std::size_t,
                                                 std::string_view row,
                                                 std::size_t offset,
                                                 int) {
    return std::get<1>(detail::chomp_quoted_string([](char) {}, delim, row, offset));
}

runtime_fixed_width_string_parser::runtime_fixed_width_string_parser(std::size_t itemsize)
    : m_itemsize(itemsize) {}

void runtime_fixed_width_string_parser::set_num_lines(std::size_t num_lines) {
    m_parsed.resize(num_lines * m_itemsize);
    m_mask.resize(num_lines);
}

std::tuple<std::size_t, bool>
runtime_fixed_width_string_parser::chomp(char delim,
                                         std::size_t ix,
                                         std::string_view row,
                                         std::size_t offset,
                                         int) {
    char* cell = &this->m_parsed[ix * m_itemsize];
    std::size_t cell_ix = 0;
    auto [mask, ret] = detail::chomp_quoted_string(
        [&](char c) {
            if (cell_ix < m_itemsize) {
                cell[cell_ix++] = c;
            }
        },
        delim,
        row,
        offset);
    this->m_mask[ix] = mask;
    return ret;
}

py::scoped_ref<> runtime_fixed_width_string_parser::move_to_python_tuple() && {
    py::scoped_ref descr(PyArray_DescrNewFromType(NPY_STRING));
    if (!descr) {
        return nullptr;
    }
    descr->elsize = m_itemsize;
    auto values = py::move_to_numpy_array(std::move(m_parsed),
                                          descr,
                                          std::array{m_mask.size()},
                                          std::array{
                                              static_cast<std::int64_t>(m_itemsize)});
    if (!values) {
        return nullptr;
    }

    auto mask_array = py::move_to_numpy_array(std::move(m_mask));
    if (!mask_array) {
        return nullptr;
    }

    return py::scoped_ref(PyTuple_Pack(2, values.get(), mask_array.get()));
}

std::tuple<std::size_t, bool> vlen_string_parser::chomp(char delim,
                                                        std::size_t ix,
                                                        std::string_view row,
                                                        std::size_t offset,
                                                        int) {
    std::string& cell = this->m_parsed[ix];
    auto [mask, ret] = detail::chomp_quoted_string([&](char c) { cell.push_back(c); },
                                                   delim,
                                                   row,
                                                   offset);
    this->m_mask[ix] = mask;
    return ret;
}

py::scoped_ref<> vlen_string_parser::move_to_python_tuple() && {
    std::vector<py::scoped_ref<>> as_ob(m_parsed.size());
    std::size_t ix = 0;
    for (const std::string& str : m_parsed) {
        if (!m_mask[ix]) {
            Py_INCREF(Py_None);
            as_ob[ix] = py::scoped_ref(Py_None);
        }
        else {
            as_ob[ix] = py::to_object(str);
            if (!as_ob[ix]) {
                return nullptr;
            }
        }
        ++ix;
    }
    auto values = py::move_to_numpy_array(std::move(as_ob));
    if (!values) {
        return nullptr;
    }

    auto mask_array = py::move_to_numpy_array(std::move(m_mask));
    if (!mask_array) {
        return nullptr;
    }

    return py::scoped_ref(PyTuple_Pack(2, values.get(), mask_array.get()));
}

std::unordered_map<std::string, detail::pcre2_code_ptr>
    runtime_format_datetime_parser::format_map;

namespace {
detail::pcre2_code_ptr jit_compile(std::string pattern) {
    int e;
    std::size_t err_offset;

    pattern.push_back('$');
    detail::pcre2_code_ptr code{
        pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()),
                      pattern.size(),
                      PCRE2_ANCHORED,
                      &e,
                      &err_offset,
                      nullptr)};
    if (!code) {
        std::array<char, 4096> buf;
        int size = pcre2_get_error_message(e,
                                           reinterpret_cast<PCRE2_UCHAR*>(buf.data()),
                                           buf.size());
        std::string_view err_msg;
        if (size < 0) {
            using namespace std::literals;

            err_msg = " <failed to get error message>"sv;
        }
        else {
            err_msg = std::string_view{buf.data(), static_cast<std::size_t>(size)};
        }
        throw util::formatted_error<std::invalid_argument>("invalid regex: \"",
                                                           pattern,
                                                           "\" reason: ",
                                                           err_msg);
    }
    if (pcre2_jit_compile(code.get(), PCRE2_JIT_COMPLETE)) {
        throw std::runtime_error{"failed to jit compile pattern"};
    }

    return code;
}
}  // namespace

namespace detail {
struct initialize_runtime_format_datetime_parser_formats {
    initialize_runtime_format_datetime_parser_formats() {
        using namespace std::literals;

        std::string year_name = "yyyy";
        std::string year_pattern = "(?<year>\\d{4})";

        std::string month_name = "mm";
        std::string month_pattern = "(?<month>\\d{1,2})";

        std::string day_name = "dd";
        std::string day_pattern = "(?<day>\\d{1,2})";

        std::array time_names = {""s, " hh:mm:ss"s, " hh:mm:ss tz"s};
        std::array time_patterns = {
            ""s,
            "( |T)(?<hour>\\d{2}):(?<min>\\d{2}):(?<sec>\\d{2})(.(?<frac_sec>\\d{1,9}))?"s,
            "( |T)(?<hour>\\d{2}):(?<min>\\d{2}):(?<sec>\\d{2})(.(?<frac_sec>\\d{1,9}))?"
            "(?<offset_sign>\\+|-)(?<offset_h>\\d{2}):?(?<offset_m>\\d{2})"s};

        std::array name_delims = {'-', '/', '.'};
        std::array pattern_delims = {"-"s, "/"s, "\\."s};

        for (const auto& [name_delim, pattern_delim] :
             py::zip(name_delims, pattern_delims)) {
            for (const auto& [time_name, time_pattern] :
                 py::zip(time_names, time_patterns)) {

                // yyyy-mm-dd
                runtime_format_datetime_parser::format_map[year_name + name_delim +
                                                           month_name + name_delim +
                                                           day_name + time_name] =
                    jit_compile(year_pattern + pattern_delim + month_pattern +
                                pattern_delim + day_pattern + time_pattern);

                // mm-dd-yyyy
                runtime_format_datetime_parser::format_map[month_name + name_delim +
                                                           day_name + name_delim +
                                                           year_name + time_name] =
                    jit_compile(month_pattern + pattern_delim + day_pattern +
                                pattern_delim + year_pattern + time_pattern);

                // dd-mm-yyyy
                runtime_format_datetime_parser::format_map[day_name + name_delim +
                                                           month_name + name_delim +
                                                           year_name + time_name] =
                    jit_compile(day_pattern + pattern_delim + month_pattern +
                                pattern_delim + year_pattern + time_pattern);
            }
        }

        // yyyymmdd
        runtime_format_datetime_parser::format_map[year_name + month_name + day_name] =
            jit_compile(year_pattern + month_pattern + day_pattern);
    }
} initialized_datetime_column_formats;
}  // namespace detail

namespace {
/** Get the group index for a regex group named `name`.

    @param code The compiled regex.
    @param name The name of the group to look up.
    @return The group index, or -1 if the group doesn't exist.
 */
int group_ix(const pcre2_code* code, const char* name) {
    int res = pcre2_substring_number_from_name(code, reinterpret_cast<PCRE2_SPTR>(name));
    if (res == PCRE2_ERROR_NOSUBSTRING) {
        return -1;
    }
    if (res < 0) {
        throw std::runtime_error{"failed to get group ix"};
    }
    return res;
}

/** Get the time since epoch for a date. This function checks that the month exists,
    and the day is the month in the given year.

    @param year The year.
    @param month The 1-indexed month.
    @param day The 1-indexed day.
    @return The time since the unix epoch.
 */
std::chrono::nanoseconds checked_time_since_epoch(int year, int month, int day) {
    if (month < 1 || month > 12) {
        throw util::formatted_error<parse_error>("invalid month: ", month);
    }
    if (day < 1 ||
        day > py::chrono::days_in_month[py::chrono::is_leapyear(year)][month - 1]) {
        throw util::formatted_error<parse_error>(
            "day is out range for month: month=", month, "; day=", day);
    }

    return py::chrono::time_since_epoch(year, month, day);
}

/** Parse hour, minute, and second information. This function checks that the values are
    in an appropriate range.

    @param hour The hour of the time.
    @param minute The minute of the time.
    @param second The second of the time.
    @return The time parsed as a duration.
 */
std::chrono::seconds checked_hour_minute_second(int hour, int minute, int second) {
    if (hour > 23) {
        throw util::formatted_error<parse_error>("hour is out of range: ", hour);
    }
    if (minute > 59) {
        throw util::formatted_error<parse_error>("minute is out of range: ", minute);
    }
    if (second > 59) {
        throw util::formatted_error<parse_error>("second is out of range: ", second);
    }

    return std::chrono::hours(hour) + std::chrono::minutes(minute) +
           std::chrono::seconds(second);
}

/** Parse a string that is known to contain only digits [0-9] and is at least `ndigits` in
    size.

    @tparam The number of digits in the string.
    @param cs The string containing only digits to parse.
    @return The parsed integer value.
 */
template<std::size_t ndigits>
int parse_int_fixed_string_unchecked(std::string_view cs) {
    static_assert(ndigits > 0, "parse_int must be at least 1 char wide");

    int result = 0;
    for (std::size_t n = 0; n < ndigits; ++n) {
        int c = cs[n] - '0';

        result *= 10;
        result += c;
    }
    return result;
}

/** Parse a string that is known to contain only digits [0-9] and is either exactly
    size 1 or size 2.

    @param cs The string containing only digits to parse.
    @return The parsed integer value.
 */
std::int64_t parse_int_flex_string_unchecked(std::string_view cs) {
    std::int64_t result = cs[0] - '0';
    if (cs.size() == 2) {
        result *= 10;
        result += cs[1] - '0';
    }
    return result;
}

/** Parse a string that is known to contain only digits [0-9].

    @param cs The string containing only digits to parse.
    @return The parsed integer value.
 */
std::int64_t parse_int_vlen_string_unchecked(std::string_view cs) {
    int result = 0;
    for (char c : cs) {
        result *= 10;
        result += c - '0';
    }
    return result;
}
}  // namespace

const pcre2_code*
runtime_format_datetime_parser::resolve_format(const std::string& format) {
    auto search = runtime_format_datetime_parser::format_map.find(format);
    if (search == runtime_format_datetime_parser::format_map.end()) {
        throw util::formatted_error<std::invalid_argument>("unknown datetime format: ",
                                                           format);
    }

    return search->second.get();
}

runtime_format_datetime_parser::runtime_format_datetime_parser(const std::string& format)
    : runtime_format_datetime_parser::runtime_format_datetime_parser(
          resolve_format(format)) {}

runtime_format_datetime_parser::runtime_format_datetime_parser(const pcre2_code* code)
    : m_code(code) {
    m_year_group = group_ix(m_code, "year");
    m_month_group = group_ix(m_code, "month");
    m_day_group = group_ix(m_code, "day");
    if (!(m_year_group > 0 && m_month_group > 0 && m_day_group > 0)) {
        throw std::invalid_argument{"missing year, month, or day component in pattern: "};
    }

    m_hour_group = group_ix(m_code, "hour");
    m_min_group = group_ix(m_code, "min");
    m_sec_group = group_ix(m_code, "sec");
    m_frac_sec_group = group_ix(m_code, "frac_sec");

    m_offset_sign_group = group_ix(m_code, "offset_sign");
    m_offset_h_group = group_ix(m_code, "offset_h");
    m_offset_m_group = group_ix(m_code, "offset_m");

    if (m_offset_sign_group > 0 && !(m_offset_h_group > 0 && m_offset_m_group > 0)) {
        throw std::invalid_argument{
            "offset_sign is provided but not offset_h or offset_m"};
    }

    if (m_hour_group > 0) {
        if (!(m_min_group > 0 && m_sec_group > 0 && m_frac_sec_group > 0)) {
            throw std::invalid_argument{"hour provided but not min, sec, or frac_sec"};
        }
        if (m_offset_sign_group > 0) {
            m_parse_func = parse_func::datetime_tz;
        }
        else {
            m_parse_func = parse_func::datetime;
        }
    }
    else {
        if (m_offset_sign_group > 0) {
            throw std::invalid_argument{"cannot have tz offset and no time"};
        }
        m_parse_func = parse_func::date;
    }
}

void runtime_format_datetime_parser::set_num_threads(int num_threads) {
    m_thread_state.resize(num_threads);
    for (thread_state& st : m_thread_state) {
        st.match_context = detail::pcre2_match_context_ptr{
            pcre2_match_context_create(nullptr)};
        st.match_data = detail::pcre2_match_data_ptr{
            pcre2_match_data_create_from_pattern(m_code, nullptr)};

        if (!(st.match_context && st.match_data)) {
            throw std::bad_alloc{};
        }
    }
}

namespace {
/** Get a string view over a matched component of a regex.

    @param sub The subject of the regex match.
    @param ovector The output vector from the match results.
    @param ix The group index.
    @return The view over the subject of the match. An empty view is returned if the
            section wasn't matched.
 */
std::string_view view_group(std::string_view sub, std::size_t* ovector, int ix) {
    std::size_t start = ovector[2 * ix];
    if (start == sub.npos) {
        return {};
    }

    std::size_t end = ovector[2 * ix + 1];
    std::size_t size = end - start;
    return sub.substr(start, size);
}
}  // namespace

py::datetime64ns runtime_format_datetime_parser::parse_date(std::size_t* ovector,
                                                            std::string_view cell) const {
    return py::datetime64ns{checked_time_since_epoch(
        parse_int_fixed_string_unchecked<4>(view_group(cell, ovector, m_year_group)),
        parse_int_flex_string_unchecked(view_group(cell, ovector, m_month_group)),
        parse_int_flex_string_unchecked(view_group(cell, ovector, m_day_group)))};
}

py::datetime64ns
runtime_format_datetime_parser::parse_datetime(std::size_t* ovector,
                                               std::string_view cell) const {
    std::chrono::nanoseconds time = checked_hour_minute_second(
        parse_int_flex_string_unchecked(view_group(cell, ovector, m_hour_group)),
        parse_int_flex_string_unchecked(view_group(cell, ovector, m_min_group)),
        parse_int_flex_string_unchecked(view_group(cell, ovector, m_sec_group)));
    std::string_view frac_sec = view_group(cell, ovector, m_frac_sec_group);
    if (frac_sec.size()) {
        std::chrono::nanoseconds ns = std::chrono::nanoseconds{
            parse_int_vlen_string_unchecked(frac_sec)};
        std::ptrdiff_t digits = frac_sec.size();
        ns *= std::pow(10, 9 - digits);
        time += ns;
    }

    return parse_date(ovector, cell) + time;
}

py::datetime64ns
runtime_format_datetime_parser::parse_datetime_tz(std::size_t* ovector,
                                                  std::string_view cell) const {
    auto offset = std::chrono::hours{parse_int_fixed_string_unchecked<2>(
                      view_group(cell, ovector, m_offset_h_group))} +
                  std::chrono::minutes{parse_int_fixed_string_unchecked<2>(
                      view_group(cell, ovector, m_offset_m_group))};

    py::datetime64ns out = parse_datetime(ovector, cell);
    if (view_group(cell, ovector, m_offset_sign_group)[0] == '+') {
        out += offset;
    }
    else {
        out -= offset;
    }
    return out;
}

std::tuple<std::size_t, bool> runtime_format_datetime_parser::chomp(char delim,
                                                                    std::size_t ix,
                                                                    std::string_view row,
                                                                    std::size_t offset,
                                                                    int thread_ix) {
    auto [cell, consumed, more] = detail::isolate_unquoted_cell(row, offset, delim);
    if (!cell.size()) {
        return {consumed, more};
    }

    thread_state& st = m_thread_state[thread_ix];

    int r = pcre2_jit_match(m_code,
                            reinterpret_cast<PCRE2_SPTR>(cell.data()),
                            cell.size(),
                            0,
                            PCRE2_ANCHORED,  // | PCRE2_ENDANCHORED, not available on
                                             // ubuntu 16.04
                            st.match_data.get(),
                            st.match_context.get());
    if (r < 0) {
        if (r == PCRE2_ERROR_NOMATCH) {
            throw util::formatted_error<parse_error>("failed to parse datetime from: \"",
                                                     cell,
                                                     '"');
        }
        throw std::runtime_error{"failed to match regex"};
    }

    std::size_t* ovector = pcre2_get_ovector_pointer(st.match_data.get());

    this->m_mask[ix] = true;
    py::datetime64ns& val = this->m_parsed[ix];
    switch (m_parse_func) {
    case parse_func::date:
        val = parse_date(ovector, cell);
        break;
    case parse_func::datetime:
        val = parse_datetime(ovector, cell);
        break;
    case parse_func::datetime_tz:
        val = parse_datetime_tz(ovector, cell);
        break;
    default:
        __builtin_unreachable();
    }

    return {consumed, more};
}

composed_column_spec::composed_column_spec(std::unique_ptr<column_spec>&& option)
    : m_option(std::move(option)) {}

column_spec::alloc_info composed_column_spec::cell_parser_alloc_info() const {
    return m_option->cell_parser_alloc_info();
}

cell_parser* composed_column_spec::emplace_cell_parser(void* addr) const {
    return m_option->emplace_cell_parser(addr);
}

float32_column::float32_column(std::string_view mode) {
    using namespace std::literals;

    if (mode == "fast"sv) {
        m_option = std::make_unique<fast_float32_column>();
    }
    else if (mode == "precise"sv) {
        m_option = std::make_unique<precise_float32_column>();
    }
    else {
        throw util::formatted_error<std::invalid_argument>(
            "unknown format: \"", mode, "\", must be one of [\"fast\", \"precise\"]");
    }
}

float64_column::float64_column(std::string_view mode) {
    using namespace std::literals;

    if (mode == "fast"sv) {
        m_option = std::make_unique<fast_float64_column>();
    }
    else if (mode == "precise"sv) {
        m_option = std::make_unique<precise_float64_column>();
    }
    else {
        throw util::formatted_error<std::invalid_argument>(
            "unknown format: \"", mode, "\", must be one of [\"fast\", \"precise\"]");
    }
}

datetime_column::datetime_column(std::string_view format)
    : m_code(runtime_format_datetime_parser::resolve_format(std::string{format})) {}

column_spec::alloc_info datetime_column::cell_parser_alloc_info() const {
    return column_spec::alloc_info::make<runtime_format_datetime_parser>();
}

cell_parser* datetime_column::emplace_cell_parser(void* addr) const {
    return new (addr) runtime_format_datetime_parser{m_code};
}

bool_column::bool_column(std::string_view fmt) {
    if (fmt == "0/1") {
        m_option = std::make_unique<simple_column_spec<bool_01_parser>>();
    }
    else if (fmt == "f/t") {
        m_option = std::make_unique<simple_column_spec<bool_ft_parser>>();
    }
    else if (fmt == "F/T") {
        m_option = std::make_unique<simple_column_spec<bool_FT_parser>>();
    }
    else if (fmt == "fF/tT") {
        m_option = std::make_unique<simple_column_spec<bool_ft_case_insensitive_parser>>();
    }
    else if (fmt == "FALSE/TRUE") {
        m_option = std::make_unique<simple_column_spec<bool_FALSE_TRUE_parser>>();
    }
    else if (fmt == "False/True") {
        m_option = std::make_unique<simple_column_spec<bool_False_True_parser>>();
    }
    else {
        throw util::formatted_error<std::invalid_argument>("invalid boolean format: ",
                                                           fmt);
    }
}

string_column::string_column(std::int64_t size) : m_size(size) {}

column_spec::alloc_info string_column::cell_parser_alloc_info() const {
    if (m_size > 0) {
        return column_spec::alloc_info::make<runtime_fixed_width_string_parser>();
    }
    else {
        return column_spec::alloc_info::make<vlen_string_parser>();
    }
}

cell_parser* string_column::emplace_cell_parser(void* addr) const {
    if (m_size > 0) {
        return new (addr)
            runtime_fixed_width_string_parser{static_cast<std::size_t>(m_size)};
    }
    else {
        return new (addr) vlen_string_parser{};
    }
}

namespace {
template<typename T>
struct destruct_but_not_free {
    void operator()(T* value) {
        if (value) {
            value->~T();
        }
    }
};

/** A unique pointer which only destructs, but doesn't free the storage. It is designed to
    be used to provide unique ownership and lifetime management of a dynamic value
    allocated in properly aligned storage for a `T`, but it doesn't manage the memory
    lifetime.
*/
template<typename T>
using destruct_only_unique_ptr = std::unique_ptr<T, destruct_but_not_free<T>>;

template<template<typename> typename ptr_type>
using parser_types = std::vector<ptr_type<cell_parser>>;

template<typename Exc, typename... Ts>
Exc position_formatted_error(std::size_t row, std::size_t col, Ts&&... msg) {
    return util::formatted_error<Exc>("line ",
                                      row,
                                      " column ",
                                      col,
                                      ": ",
                                      std::forward<Ts>(msg)...);
}

/** Parse a single row and store the values in the vectors.
    @param row The row index.
    @param data The view over the row to parse.
    @param parsers The cell parsers.
 */
template<template<typename> typename ptr_type>
void parse_row(std::size_t row,
               char delim,
               std::string_view data,
               parser_types<ptr_type>& parsers,
               int thread_ix) {

    std::size_t col = 0;
    std::size_t consumed = 0;
    bool more = true;

    for (auto& parser : parsers) {
        if (!more) {
            throw util::formatted_error<parse_error>("line ",
                                                     row + 2,
                                                     ": less columns than expected, got ",
                                                     col,
                                                     " but expected ",
                                                     parsers.size());
        }
        try {
            auto [new_consumed,
                  new_more] = parser->chomp(delim, row, data, consumed, thread_ix);
            consumed += new_consumed;
            more = new_more;
        }
        catch (const std::exception& e) {
            throw position_formatted_error<parse_error>(row + 2, col, e.what());
        }

        ++col;
    }

    if (consumed != data.size()) {
        throw util::formatted_error<parse_error>(
            "line ", row + 2, ": more columns than expected, expected ", parsers.size());
    }
}

template<template<typename> typename ptr_type>
void parse_lines(std::string_view data,
                 char delim,
                 std::size_t data_offset,
                 const std::vector<std::size_t>& line_sizes,
                 std::size_t line_end_size,
                 std::size_t offset,
                 parser_types<ptr_type>& parsers,
                 int thread_ix) {
    std::size_t ix = offset;
    for (const auto& size : line_sizes) {
        auto row = data.substr(data_offset, size);
        LIBPY_CSV_PREFETCH(row.data() + size, 0, 0);
        LIBPY_CSV_PREFETCH(row.data() + size + l1dcache_line_size, 0, 0);
        LIBPY_CSV_PREFETCH(row.data() + size + 2 * l1dcache_line_size, 0, 0);
        parse_row<ptr_type>(ix, delim, row, parsers, thread_ix);
        data_offset += size + line_end_size;
        ++ix;
    }
}

template<template<typename> typename ptr_type>
void parse_lines_worker(std::mutex* exception_mutex,
                        std::vector<std::exception_ptr>* exceptions,
                        std::string_view data,
                        char delim,
                        const std::size_t data_offset,
                        const std::vector<std::size_t>* line_sizes,
                        std::size_t line_end_size,
                        std::size_t offset,
                        parser_types<ptr_type>* parsers,
                        int thread_ix) {
    try {
        parse_lines<ptr_type>(data,
                              delim,
                              data_offset,
                              *line_sizes,
                              line_end_size,
                              offset,
                              *parsers,
                              thread_ix);
    }
    catch (const std::exception&) {
        std::lock_guard<std::mutex> guard(*exception_mutex);
        exceptions->emplace_back(std::current_exception());
    }
}

/** Split a buffer into a vector of lines.

    The lines are to be processed in order, so only the line lengths are recorded. The
    line lengths do *not* include the length of the line delimiter.

    The in-out parameter `pos_ptr` indicates the index into `data` to start counting.
    To ensure consistent counting multithreaded counting, line searching will begin
    by looking backwards from `*pos_ptr` to find the previous line delimiter, then lines
    will be counted until the end of the line is > `end_ix`. `pos_ptr` will be overwritten
    with the new starting location.

    @param lines The output vector for the lines.
    @param data The buffer to cut into lines.
    @param pos_ptr The in-out param indicating where to start searching for lines. This
           will be overwritten to be the starting point from which to interpret `lines`.
    @param end_ix The end ix for searching for lines.
    @param line_ending The line ending to split on.
    @param handle_tail Should we count anything after the last newline? This should be
           true for the invocation that is counting up to the end of the buffer.
 */
void split_into_lines_loop(std::vector<std::size_t>& lines,
                           std::string_view data,
                           std::string_view::size_type* pos_ptr,
                           std::string_view::size_type end_ix,
                           std::string_view line_ending,
                           bool handle_tail) {
    // Crawl the start back to the beginning of the line that we begin inside. This is
    // done by searching for the newline strictly before `*pos_ptr`.
    auto line_start = data.substr(0, *pos_ptr).rfind(line_ending);
    if (line_start == std::string_view::npos) {
        // there is no newline prior to `*pos_ptr`, so we must be in the middle
        // of the first line, set the `*pos_ptr` back to 0.
        *pos_ptr = 0;
    }
    else {
        // `pos_ptr` was pointing into the middle of the line that has a newline
        // before it, set `pos_ptr` back to the start of the line
        *pos_ptr = line_start + line_ending.size();
    }

    auto pos = *pos_ptr;
    std::string_view::size_type end;
    while ((end = data.find(line_ending, pos)) != std::string_view::npos) {
        if (end > end_ix) {
            return;
        }

        auto size = end - pos;
        lines.emplace_back(size);

        // advance past line ending
        pos = end + line_ending.size();

        LIBPY_CSV_PREFETCH(data.data() + end + size, 0, 0);
        LIBPY_CSV_PREFETCH(data.data() + end + size + l1dcache_line_size, 0, 0);
    }

    if (handle_tail and pos < end_ix) {
        // add any data after the last newline if there is anything to add
        lines.emplace_back(end_ix - pos);
    }
}

void split_into_lines_worker(std::mutex* exception_mutex,
                             std::vector<std::exception_ptr>* exceptions,
                             std::vector<std::size_t>* lines,
                             std::string_view data,
                             std::string_view::size_type* pos,
                             std::string_view::size_type end_ix,
                             std::string_view line_ending,
                             bool handle_tail) {
    try {
        split_into_lines_loop(*lines, data, pos, end_ix, line_ending, handle_tail);
    }
    catch (const std::exception&) {
        std::lock_guard<std::mutex> guard(*exception_mutex);
        exceptions->emplace_back(std::current_exception());
    }
}

std::tuple<std::vector<std::size_t>, std::vector<std::vector<std::size_t>>>
split_into_lines(std::string_view data,
                 std::string_view line_ending,
                 std::size_t num_columns,
                 std::size_t skip_rows,
                 std::size_t num_threads) {
    if (num_threads == 0) {
        num_threads = 1;
    }

    std::size_t group_size = data.size() / num_threads + 1;
    if (group_size < min_split_lines_bytes_size) {
        // not really worth breaking this file up
        num_threads = 1;
    }

    std::vector<std::vector<std::size_t>> lines_per_thread(num_threads);
    std::vector<std::size_t> thread_starts(num_threads);
    for (auto& lines : lines_per_thread) {
        // assume that each column will take about 5 bytes of data on average
        lines.reserve(std::ceil(group_size / (5.0 * num_columns)));
    }

    // The current position into the input.
    std::string_view::size_type pos = 0;
    // The index of the next newline.
    std::string_view::size_type end;

    // optionally skip some rows
    for (std::size_t n = 0; n < skip_rows; ++n) {
        if ((end = data.find(line_ending, pos)) == std::string_view::npos) {
            break;
        }
        // advance past line ending
        pos = end + line_ending.size();
    }
    data = data.substr(pos);
    pos = 0;

    if (num_threads == 1) {
        split_into_lines_loop(lines_per_thread[0],
                              data,
                              &pos,
                              data.size(),
                              line_ending,
                              /* handle_tail */ true);
        thread_starts[0] = pos;
    }
    else {
        std::mutex exception_mutex;
        std::vector<std::exception_ptr> exceptions;

        std::vector<std::thread> threads;
        for (std::size_t n = 0; n < num_threads; ++n) {
            thread_starts[n] = pos + n * group_size;
            threads.emplace_back(
                std::thread(split_into_lines_worker,
                            &exception_mutex,
                            &exceptions,
                            &lines_per_thread[n],
                            data,
                            &thread_starts[n],
                            std::min(thread_starts[n] + group_size, data.size()),
                            line_ending,
                            /* handle_thread */ n == num_threads - 1));
        }

        for (auto& thread : threads) {
            thread.join();
        }

        for (auto& e : exceptions) {
            std::rethrow_exception(e);
        }
    }

    return {std::move(thread_starts), std::move(lines_per_thread)};
}

/** Parse a full CSV given the header and the types of the columns.
    @param data The CSV to parse.
    @param parsers The cell parsers.
    @param line_ending The string to split lines on.
 */
template<template<typename> typename ptr_type>
void parse_from_header(std::string_view data,
                       parser_types<ptr_type>& parsers,
                       char delimiter,
                       std::string_view line_ending,
                       std::size_t num_threads,
                       std::size_t skip_rows = 0) {
    auto [thread_starts, line_sizes_per_thread] =
        split_into_lines(data, line_ending, parsers.size(), skip_rows, num_threads);

    std::size_t lines = std::accumulate(line_sizes_per_thread.begin(),
                                        line_sizes_per_thread.end(),
                                        0,
                                        [](auto sum, const auto& vec) {
                                            return sum + vec.size();
                                        });

    for (auto& parser : parsers) {
        parser->set_num_lines(lines);
        parser->set_num_threads((line_sizes_per_thread.size() == 1) ? 1 : num_threads);
    }

    if (line_sizes_per_thread.size() == 1) {
        parse_lines<ptr_type>(data,
                              delimiter,
                              thread_starts[0],
                              line_sizes_per_thread[0],
                              line_ending.size(),
                              0,
                              parsers,
                              0);
    }
    else {
        std::mutex exception_mutex;
        std::vector<std::exception_ptr> exceptions;

        std::vector<std::thread> threads;
        std::size_t start = 0;
        for (std::size_t n = 0; n < num_threads; ++n) {
            threads.emplace_back(std::thread(parse_lines_worker<ptr_type>,
                                             &exception_mutex,
                                             &exceptions,
                                             data,
                                             delimiter,
                                             thread_starts[n],
                                             &line_sizes_per_thread[n],
                                             line_ending.size(),
                                             start,
                                             &parsers,
                                             n));
            start += line_sizes_per_thread[n].size();
        }

        for (auto& thread : threads) {
            thread.join();
        }

        for (auto& e : exceptions) {
            std::rethrow_exception(e);
        }
    }
}

template<typename T>
auto lookup_type() {
    auto out = py::autoclass<T>::lookup_type();
    if (!out) {
        throw py::exception(PyExc_ValueError,
                            py::util::type_name<T>().get(),
                            " was not autoclassed");
    }
    return out;
}

template<typename... Ts>
struct unbox_spec_helper;

template<typename Head, typename... Tail>
struct unbox_spec_helper<Head, Tail...> {
    static column_spec* f(PyObject* value) {
        if (PyObject_IsInstance(value,
                                reinterpret_cast<PyObject*>(lookup_type<Head>().get()))) {
            return &py::autoclass<Head>::unbox(value);
        }
        return unbox_spec_helper<Tail...>::f(value);
    }
};

template<>
struct unbox_spec_helper<> {
    static column_spec* f(PyObject* value) {
        throw py::exception(PyExc_ValueError,
                            "unknown column spec type:",
                            Py_TYPE(value));
    }
};

column_spec* unbox_spec(PyObject* spec) {
    return unbox_spec_helper<skip_column,
                             int8_column,
                             int16_column,
                             int32_column,
                             int64_column,
                             float32_column,
                             float64_column,
                             datetime_column,
                             bool_column,
                             string_column>::f(spec);
}

template<typename InitColumn, typename Init>
std::string_view parse_header(std::string_view data,
                              char delimiter,
                              std::string_view line_ending,
                              Init&& init,
                              InitColumn&& init_column) {
    auto line_end = data.find(line_ending, 0);
    auto line = data.substr(0, line_end);

    header_parser header_parser;
    for (auto [consumed, more] = std::make_tuple(0, true); more;) {
        auto [new_consumed,
              new_more] = header_parser.chomp(delimiter, 0, line, consumed, 0);
        consumed += new_consumed;
        more = new_more;
    }

    init(header_parser.parsed().size());

    std::unordered_set<std::string> column_names;
    std::size_t ix = 0;
    for (const auto& cell : header_parser.parsed()) {
        if (column_names.count(cell)) {
            throw util::formatted_error<parse_error>("column name duplicated: ", cell);
        }
        column_names.emplace(cell);
        init_column(ix, cell);
        ++ix;
    };

    auto start = std::min(line.size() + line_ending.size(), data.size());
    return data.substr(start);
}

class parsers {
private:
    std::unique_ptr<std::byte[]> m_storage;
    parser_types<destruct_only_unique_ptr> m_ptrs;

public:
    parsers(std::unique_ptr<std::byte[]>&& storage,
            parser_types<destruct_only_unique_ptr>&& ptrs)
        : m_storage(std::move(storage)), m_ptrs(std::move(ptrs)) {}
    parser_types<destruct_only_unique_ptr>& ptrs() {
        return m_ptrs;
    }

    ~parsers() {
        // execute the destructors before we deallocate the storage.
        m_ptrs.clear();
    }
};

parsers allocate_parsers(const std::vector<column_spec*>& specs) {
    std::size_t storage_size = 0;
    std::vector<std::size_t> offsets(specs.size());
    std::size_t ix = 0;
    for (const column_spec* spec : specs) {
        column_spec::alloc_info alloc_info = spec->cell_parser_alloc_info();
        storage_size += storage_size % alloc_info.align;
        offsets[ix] = storage_size;
        storage_size += alloc_info.size;
        ++ix;
    }

    std::unique_ptr<std::byte[]> cell_parser_storage(new (std::align_val_t{
        specs[0]->cell_parser_alloc_info().align}) std::byte[storage_size]);
    if (!cell_parser_storage) {
        throw std::bad_alloc{};
    }
    parser_types<destruct_only_unique_ptr> semantic_ptrs(specs.size());

    ix = 0;
    for (const column_spec* spec : specs) {
        semantic_ptrs[ix] = destruct_only_unique_ptr<cell_parser>(
            spec->emplace_cell_parser(&cell_parser_storage[offsets[ix]]));
        ++ix;
    }

    return parsers{std::move(cell_parser_storage), std::move(semantic_ptrs)};
}

void verify_column_specs_dict(PyObject* specs, std::vector<py::scoped_ref<>>& header) {
    py::scoped_ref expected_keys(PySet_New(specs));
    if (!expected_keys) {
        throw py::exception();
    }
    py::scoped_ref actual_keys = py::to_object(header);
    if (!actual_keys) {
        throw py::exception();
    }

    py::scoped_ref actual_keys_set(PySet_New(actual_keys.get()));
    if (!actual_keys_set) {
        throw py::exception();
    }

    py::scoped_ref diff(PyNumber_Subtract(expected_keys.get(), actual_keys_set.get()));
    if (!diff) {
        throw py::exception();
    }
    if (PySet_GET_SIZE(diff.get())) {
        py::scoped_ref as_list(PySequence_List(diff.get()));
        if (PyList_Sort(as_list.get()) < 0) {
            throw py::exception();
        }
        throw py::exception(PyExc_ValueError,
                            "column spec keys not present in header: ",
                            as_list,
                            "\nheader: ",
                            actual_keys);
    }
}
}  // namespace

void parse(std::string_view data,
           const std::unordered_map<std::string, std::shared_ptr<cell_parser>>& types,
           char delimiter,
           std::string_view line_ending,
           std::size_t num_threads) {
    parser_types<std::shared_ptr> parsers;

    auto init = [&](std::size_t num_columns) { parsers.resize(num_columns); };

    auto init_parser = [&](std::size_t ix, const auto& cell) {
        auto search = types.find(cell);
        if (search == types.end()) {
            parsers[ix] = std::make_shared<skip_parser>();
        }
        else {
            parsers[ix] = search->second;
        }
    };

    std::string_view to_parse =
        parse_header(data, delimiter, line_ending, init, init_parser);

    parse_from_header(to_parse, parsers, delimiter, line_ending, num_threads);
}

using namespace py::cs::literals;

PyObject*
py_parse(PyObject*,
         std::string_view data,
         py::arg::keyword<decltype("column_specs"_cs), PyObject*> column_specs,
         py::arg::keyword<decltype("delimiter"_cs), char> delimiter,
         py::arg::keyword<decltype("line_ending"_cs), std::string_view> line_ending,
         py::arg::keyword<decltype("num_threads"_cs), std::size_t> num_threads) {
    Py_ssize_t num_specified_columns = PyDict_Size(column_specs.get());
    if (num_specified_columns < 0) {
        // use `PyDict_Size` to ensure this is a dict with a reasonable error message
        return nullptr;
    }

    if (num_specified_columns == 0) {
        // empty dict, data doesn't matter
        return PyDict_New();
    }

    skip_column skip;
    std::vector<column_spec*> specs;
    std::vector<py::scoped_ref<>> header;

    auto init = [&](std::size_t num_cols) {
        specs.resize(num_cols);
        header.resize(num_cols);
    };

    auto init_column = [&](std::size_t ix, const auto& cell) {
        auto& cell_ob = header[ix] = py::to_object(cell);

        PyObject* spec = PyDict_GetItem(column_specs.get(), cell_ob.get());
        if (spec) {
            specs[ix] = unbox_spec(spec);
        }
        else {
            specs[ix] = &skip;
        }
    };

    std::string_view to_parse =
        parse_header(data, delimiter.get(), line_ending.get(), init, init_column);

    auto parsers = allocate_parsers(specs);

    verify_column_specs_dict(column_specs.get(), header);
    parse_from_header<destruct_only_unique_ptr>(to_parse,
                                                parsers.ptrs(),
                                                delimiter.get(),
                                                line_ending.get(),
                                                num_threads.get());

    py::scoped_ref out(PyDict_New());
    if (!out) {
        return nullptr;
    }

    for (std::size_t ix = 0; ix < header.size(); ++ix) {
        auto& name = header[ix];
        auto& parser = parsers.ptrs()[ix];

        auto value = std::move(*parser).move_to_python_tuple();
        if (!value) {
            return nullptr;
        }
        if (value.get() == Py_None) {
            continue;
        }

        if (PyDict_SetItem(out.get(), name.get(), value.get())) {
            return nullptr;
        }
    }

    return std::move(out).escape();
}

bool add_parser_pytypes(PyObject* module) {
    try {
        py::scoped_ref<> modname_ob = py::getattr(module, "__name__");
        if (!modname_ob) {
            return true;
        }
        std::string modname{py::util::pystring_to_string_view(modname_ob)};
        static std::vector types = {
            py::autoclass<skip_column>(modname + ".Skip").new_<>().type(),
            py::autoclass<int8_column>(modname + ".Int8").new_<>().type(),
            py::autoclass<int16_column>(modname + ".Int16").new_<>().type(),
            py::autoclass<int32_column>(modname + ".Int32").new_<>().type(),
            py::autoclass<int64_column>(modname + ".Int64").new_<>().type(),
            py::autoclass<float32_column>(modname + ".Float32")
                .new_<std::string_view>()
                .type(),
            py::autoclass<float64_column>(modname + ".Float64")
                .new_<std::string_view>()
                .type(),
            py::autoclass<datetime_column>(modname + ".DateTime")
                .new_<std::string_view>()
                .type(),
            py::autoclass<bool_column>(modname + ".Bool").new_<std::string_view>().type(),
            py::autoclass<string_column>(modname + ".String").new_<std::int64_t>().type(),
        };

        for (const auto& type : types) {
            const char* name = std::strrchr(type->tp_name, '.');
            if (!name) {
                py::raise(PyExc_AssertionError)
                    << "name " << type->tp_name << " is not in a module";
                return true;
            }
            name += 1;  // advance past the `.` char

            if (PyObject_SetAttrString(module, name, static_cast<PyObject*>(type))) {
                return true;
            }
        }
    }
    catch (...) {
        return true;
    }
    return false;
}
}  // namespace py::csv::parser
