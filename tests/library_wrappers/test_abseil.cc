#include "libpy/library_wrappers/abseil.h"

#include "test_utils.h"

namespace test_abseil {

class abseil_to_object : public with_python_interpreter {};

TEST_F(abseil_to_object, btree_map) {
    auto map = absl::btree_map<std::string, bool>();
    py_test::test_map_to_object_impl(map);
}

TEST_F(abseil_to_object, btree_set) {
    auto filler = py_test::examples<std::string>();
    auto a = absl::btree_set<std::string>(filler.begin(), filler.end());
    py_test::test_set_to_object_impl(a);
}

}  // namespace test_abseil
