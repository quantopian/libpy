#include <limits>
#include <stdexcept>

#include "gtest/gtest.h"

#include "libpy/datetime64.h"
#include "libpy/itertools.h"
#include "libpy/library_wrappers/sparsehash.h"

#include "test_utils.h"

namespace test_sparsehash {

using namespace std::literals;
using namespace py::cs::literals;

class sparsehash_to_object : public with_python_interpreter {};

TEST_F(sparsehash_to_object, sparse_hash_map) {
    // NOTE: This test takes a long time to compile (about a .5s per entry in this
    // tuple). This is just enough coverage to test all three of our hash table types,
    // and a few important key/value types.
    auto map = google::sparse_hash_map<std::string, bool>();

    py_test::test_map_to_object_impl(map);
}

TEST_F(sparsehash_to_object, dense_hash_map) {
    auto map = google::dense_hash_map<std::string, bool>();
    map.set_empty_key("the_empty_key"s);

    py_test::test_map_to_object_impl(map);
}

TEST_F(sparsehash_to_object, sparse_hash_set) {
    auto filler = py_test::examples<std::string>();
    auto a = google::sparse_hash_set<std::string>(filler.begin(), filler.end());
    py_test::test_set_to_object_impl(a);
}

TEST_F(sparsehash_to_object, dense_hash_set) {
    auto filler = py_test::examples<std::string>();
    auto a = google::dense_hash_set<std::string>(filler.begin(),
                                                 filler.end(),
                                                 "the_empty_key"s);
    py_test::test_set_to_object_impl(a);
}

TEST(dense_hash_map, invalid_empty_key) {
    using double_key = py::dense_hash_map<double, int>;
    EXPECT_THROW((double_key{std::numeric_limits<double>::quiet_NaN()}),
                 std::invalid_argument);
    EXPECT_THROW((double_key{std::numeric_limits<double>::quiet_NaN(), 10}),
                 std::invalid_argument);

    using float_key = py::dense_hash_map<float, int>;
    EXPECT_THROW((float_key{std::numeric_limits<float>::quiet_NaN()}),
                 std::invalid_argument);
    EXPECT_THROW((float_key{std::numeric_limits<float>::quiet_NaN(), 10}),
                 std::invalid_argument);

    using M8_key = py::dense_hash_map<py::datetime64ns, int>;
    EXPECT_THROW((M8_key{py::datetime64ns::nat()}), std::invalid_argument);
    EXPECT_THROW((M8_key{py::datetime64ns::nat(), 10}), std::invalid_argument);
}

TEST(dense_hash_set, invalid_empty_key) {
    using double_key = py::dense_hash_set<double>;
    EXPECT_THROW((double_key{std::numeric_limits<double>::quiet_NaN()}),
                 std::invalid_argument);
    EXPECT_THROW((double_key{std::numeric_limits<double>::quiet_NaN(), 10}),
                 std::invalid_argument);

    using float_key = py::dense_hash_set<float>;
    EXPECT_THROW((float_key{std::numeric_limits<float>::quiet_NaN()}),
                 std::invalid_argument);
    EXPECT_THROW((float_key{std::numeric_limits<float>::quiet_NaN(), 10}),
                 std::invalid_argument);

    using M8_key = py::dense_hash_set<py::datetime64ns>;
    EXPECT_THROW((M8_key{py::datetime64ns::nat()}), std::invalid_argument);
    EXPECT_THROW((M8_key{py::datetime64ns::nat(), 10}), std::invalid_argument);
}

}  // namespace test_sparsehash
