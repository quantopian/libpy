#include <array>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"

#include "libpy/char_sequence.h"
#include "libpy/csv.h"
#include "libpy/datetime64.h"

namespace test_csv {
const char* failing_string = "123x";

TEST(fast_strtol, int8) {
    std::array<const char*, 7> strings = {"0", "1", "64", "127", "-1", "-64", "-127"};
    std::array<std::int8_t, 7> expected_values = {0, 1, 64, 127, -1, -64, -127};

    for (const auto& [input, expected] : py::zip(strings, expected_values)) {
        const char* out;
        auto actual = py::csv::parser::fast_strtol<std::int8_t>(input, &out);
        EXPECT_EQ(out - input, static_cast<std::ptrdiff_t>(std::strlen(input)));
        EXPECT_EQ(actual, expected);
    }
}

TEST(fast_strtol, int8_overflow) {
    std::array<const char*, 4> strings = {"256", "-256", "10000", "-10000"};

    for (const char* input : strings) {
        const char* out;
        EXPECT_THROW(py::csv::parser::fast_strtol<std::int8_t>(input, &out),
                     std::overflow_error)
            << "input=" << input;
    }
}

TEST(fast_strtol, int16) {
    std::array<const char*, 7> strings =
        {"0", "1", "10000", "32767", "-1", "-10000", "-32767"};
    std::array<std::int16_t, 7> expected_values =
        {0, 1, 10000, 32767, -1, -10000, -32767};

    for (const auto& [input, expected] : py::zip(strings, expected_values)) {
        const char* out;
        auto actual = py::csv::parser::fast_strtol<std::int16_t>(input, &out);
        EXPECT_EQ(out - input, static_cast<std::ptrdiff_t>(std::strlen(input)));
        EXPECT_EQ(actual, expected);
    }
}

TEST(fast_strtol, int16_overflow) {
    std::array<const char*, 4> strings = {"32768", "-32768", "100000", "-100000"};

    for (const char* input : strings) {
        const char* out;
        EXPECT_THROW(py::csv::parser::fast_strtol<std::int16_t>(input, &out),
                     std::overflow_error)
            << "input=" << input;
    }
}

TEST(fast_strtol, int32) {
    std::array<const char*, 7> strings =
        {"0", "1", "1000000", "2147483647", "-1", "-1000000", "-2147483647"};
    std::array<std::int32_t, 7> expected_values =
        {0, 1, 1000000, 2147483647, -1, -1000000, -2147483647};

    for (const auto& [input, expected] : py::zip(strings, expected_values)) {
        const char* out;
        auto actual = py::csv::parser::fast_strtol<std::int32_t>(input, &out);
        EXPECT_EQ(out - input, static_cast<std::ptrdiff_t>(std::strlen(input)));
        EXPECT_EQ(actual, expected);
    }
}

TEST(fast_strtol, int32_overflow) {
    std::array<const char*, 4> strings = {"2147483648",
                                          "-2147483648",
                                          "10000000000000",
                                          "-10000000000000"};

    for (const char* input : strings) {
        const char* out;
        EXPECT_THROW(py::csv::parser::fast_strtol<std::int32_t>(input, &out),
                     std::overflow_error)
            << "input=" << input;
    }
}

TEST(fast_strtol, int64) {
    std::array<const char*, 7> strings = {"0",
                                          "1",
                                          "1000000000000",
                                          "9223372036854775807",
                                          "-1",
                                          "-1000000000000",
                                          "-9223372036854775807"};
    std::array<std::int64_t, 7> expected_values = {0,
                                                   1,
                                                   1000000000000,
                                                   9223372036854775807,
                                                   -1,
                                                   -1000000000000,
                                                   -9223372036854775807};

    for (const auto& [input, expected] : py::zip(strings, expected_values)) {
        const char* out;
        auto actual = py::csv::parser::fast_strtol<std::int64_t>(input, &out);
        EXPECT_EQ(out - input, static_cast<std::ptrdiff_t>(std::strlen(input)));
        EXPECT_EQ(actual, expected);
    }
}

TEST(fast_strtol, int64_overflow) {
    std::array<const char*, 4> strings = {"9223372036854775808",
                                          "-9223372036854775808",
                                          "1000000000000000000000000000",
                                          "-1000000000000000000000000000"};

    for (const char* input : strings) {
        const char* out;
        EXPECT_THROW(py::csv::parser::fast_strtol<std::int64_t>(input, &out),
                     std::overflow_error)
            << "input=" << input;
    }
}

template<typename T>
class fast_strtol_errors : public testing::Test {};
TYPED_TEST_CASE_P(fast_strtol_errors);

TYPED_TEST_P(fast_strtol_errors, invalid_string) {
    std::array<const char*, 4> strings = {"x", "1x", "12x", "123x"};
    std::array<TypeParam, 4> expected_values = {0, 1, 12, 123};
    std::array<std::ptrdiff_t, 4> expected_lengths = {0, 1, 2, 3};

    for (const auto& [input, expected_value, expected_lenth] :
         py::zip(strings, expected_values, expected_lengths)) {

        const char* out;
        auto actual = py::csv::parser::fast_strtol<TypeParam>(input, &out);

        EXPECT_EQ(out - input, expected_lenth);
        EXPECT_EQ(actual, expected_value);
    }
}
REGISTER_TYPED_TEST_CASE_P(fast_strtol_errors, invalid_string);

using int_types = testing::Types<std::int8_t, std::int16_t, std::int32_t, std::int64_t>;
INSTANTIATE_TYPED_TEST_CASE_P(typed_, fast_strtol_errors, int_types);

struct csv_params {
    char delim;
    const char* line_sep;
};

std::ostream& operator<<(std::ostream& s, const csv_params& p) {
    return s << "{delim='" << p.delim << "', line_sep=\"" << p.line_sep << "\"}";
}

class parse_csv : public testing::TestWithParam<csv_params> {};

namespace detail {
template<std::size_t counter, std::size_t ncol, typename T, typename... Ts>
void build_csv_helper(const csv_params& params,
                      std::stringstream& builder,
                      const T& entry,
                      const Ts&... entries) {
    builder << entry;

    if (counter == ncol - 1) {
        builder << params.line_sep;
    }
    else {
        builder << params.delim;
    }

    if constexpr (sizeof...(entries) > 0) {
        build_csv_helper<(counter + 1) % ncol, ncol>(params, builder, entries...);
    }
}
}  // namespace detail

template<std::size_t ncols, typename... Ts>
std::string build_csv(const csv_params& params, const Ts&... entries) {
    static_assert(sizeof...(entries) % ncols == 0, "mismatched shape of csv entries");

    std::stringstream builder;
    detail::build_csv_helper<0, ncols>(params, builder, entries...);
    return builder.str();
}

TEST_P(parse_csv, parse_simple) {
    csv_params params = GetParam();
    // clang-format off
    std::string csv_content = build_csv<5>(
        params,
        "int64", "float64", "date", "datetime", "string",
        1, 2.5, "2014-01-01", "2015-01-01 13:30:30", "ayy",
        2, 3.5, "2014-01-02", "2015-01-02 14:40:40", "lmao",
        3, 4.5, "2014-01-03", "2015-01-03 15:50:50", "fam");
    // clang-format on

    auto int64_parser = std::make_shared<py::csv::parser::int64_parser>();
    auto float64_parser = std::make_shared<py::csv::parser::precise_float64_parser>();
    auto date_parser = std::make_shared<py::csv::parser::runtime_format_datetime_parser>(
        "yyyy-mm-dd");
    auto runtime_format_datetime_parser =
        std::make_shared<py::csv::parser::runtime_format_datetime_parser>(
            "yyyy-mm-dd hh:mm:ss");
    auto string_parser =
        std::make_shared<py::csv::parser::fixed_width_string_parser<4>>();

    std::unordered_map<std::string, std::shared_ptr<py::csv::parser::cell_parser>> cols =
        {{"int64", std::static_pointer_cast<py::csv::parser::cell_parser>(int64_parser)},
         {"float64",
          std::static_pointer_cast<py::csv::parser::cell_parser>(float64_parser)},
         {"date", std::static_pointer_cast<py::csv::parser::cell_parser>(date_parser)},
         {"datetime",
          std::static_pointer_cast<py::csv::parser::cell_parser>(
              runtime_format_datetime_parser)},
         {"string",
          std::static_pointer_cast<py::csv::parser::cell_parser>(string_parser)}};

    py::csv::parse(csv_content.data(), cols, params.delim, params.line_sep, 0);

    using s = py::datetime64<py::chrono::s>;
    using namespace py::cs::literals;

    std::vector<py::py_bool> expected_mask = {true, true, true};
    std::vector<std::int64_t> expected_int64 = {1, 2, 3};
    std::vector<double> expected_float64 = {2.5, 3.5, 4.5};
    std::vector<py::datetime64ns> expected_date = {s(1388534400),
                                                   s(1388620800),
                                                   s(1388707200)};
    std::vector<py::datetime64ns> expected_datetime = {s(1420119030),
                                                       s(1420209640),
                                                       s(1420300250)};
    std::vector<std::array<char, 4>> expected_string = {"ayy\0"_arr,
                                                        "lmao"_arr,
                                                        "fam\0"_arr};

    auto [actual_int64, actual_int64_mask] = std::move(*int64_parser).move_to_tuple();
    EXPECT_EQ(actual_int64, expected_int64);
    EXPECT_EQ(actual_int64_mask, expected_mask);

    auto [actual_float64,
          actual_float64_mask] = std::move(*float64_parser).move_to_tuple();
    EXPECT_EQ(actual_float64, expected_float64);
    EXPECT_EQ(actual_float64_mask, expected_mask);

    auto [actual_date, actual_date_mask] = std::move(*date_parser).move_to_tuple();
    EXPECT_EQ(actual_date, expected_date);
    EXPECT_EQ(actual_date_mask, expected_mask);

    auto [actual_datetime, actual_datetime_mask] =
        std::move(*runtime_format_datetime_parser).move_to_tuple();
    EXPECT_EQ(actual_datetime, expected_datetime);
    EXPECT_EQ(actual_datetime_mask, expected_mask);

    auto [actual_string, actual_string_mask] = std::move(*string_parser).move_to_tuple();
    EXPECT_EQ(actual_string, expected_string);
    EXPECT_EQ(actual_string_mask, expected_mask);
}

INSTANTIATE_TEST_CASE_P(csv_with_params,
                        parse_csv,
                        testing::Values(csv_params{',', "\n"},
                                        csv_params{',', "\r\n"},
                                        csv_params{'|', "\n"},
                                        csv_params{'|', "\r\n"}));
}  // namespace test_csv
