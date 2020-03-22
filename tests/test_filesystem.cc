#include "test_utils.h"

#include "libpy/filesystem.h"

namespace test_filesystem {
using namespace std::literals;
class filesystem : public with_python_interpreter {};

TEST_F(filesystem, to_path) {
#if PY_VERSION_HEX >= 0x0306
    py::scoped_ref ns = RUN_PYTHON(R"(
        class C(object):
            def __fspath__(self):
                return "/tmp/"

        ob = C()
        good = "/tmp/"
        goodb = b"/tmp/"
        bad = 4
    )");
    ASSERT_TRUE(ns);
    std::filesystem::path test_path = "/tmp/";

    PyObject* ob = PyDict_GetItemString(ns.get(), "ob");
    ASSERT_TRUE(ob);
    EXPECT_EQ(test_path, py::filesystem::fs_path(ob));

    PyObject* good = PyDict_GetItemString(ns.get(), "good");
    ASSERT_TRUE(good);
    EXPECT_EQ(test_path, py::filesystem::fs_path(good));

    PyObject* goodb = PyDict_GetItemString(ns.get(), "goodb");
    ASSERT_TRUE(goodb);
    EXPECT_EQ(test_path, py::filesystem::fs_path(goodb));

    PyObject* bad = PyDict_GetItemString(ns.get(), "bad");
    ASSERT_TRUE(bad);
    EXPECT_THROW(auto bad_path = py::filesystem::fs_path(bad), py::exception);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "expected str, bytes or os.PathLike object, not int");
    PyErr_Clear();
#endif
}

}  // namespace test_filesystem
