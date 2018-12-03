#include <array>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"

#include "libpy/char_sequence.h"
#include "libpy/csv.h"
#include "libpy/datetime64.h"

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

    if constexpr (sizeof...(entries)) {
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

    auto int64_parser = std::make_shared<py::csv::typed_cell_parser<std::int64_t>>();
    auto float64_parser = std::make_shared<py::csv::typed_cell_parser<double>>();
    auto date_parser =
        std::make_shared<py::csv::typed_cell_parser<py::datetime64<py::chrono::D>>>();
    auto datetime_parser =
        std::make_shared<py::csv::typed_cell_parser<py::datetime64<py::chrono::s>>>();
    auto string_parser =
        std::make_shared<py::csv::typed_cell_parser<std::array<char, 4>>>();

    std::unordered_map<std::string, std::shared_ptr<py::csv::cell_parser>> cols =
        {{"int64", std::static_pointer_cast<py::csv::cell_parser>(int64_parser)},
         {"float64", std::static_pointer_cast<py::csv::cell_parser>(float64_parser)},
         {"date", std::static_pointer_cast<py::csv::cell_parser>(date_parser)},
         {"datetime", std::static_pointer_cast<py::csv::cell_parser>(datetime_parser)},
         {"string", std::static_pointer_cast<py::csv::cell_parser>(string_parser)}};

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

    auto [actual_datetime,
          actual_datetime_mask] = std::move(*datetime_parser).move_to_tuple();
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
