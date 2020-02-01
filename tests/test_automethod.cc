#include <forward_list>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "libpy/automethod.h"
#include "libpy/call_function.h"
#include "libpy/exception.h"
#include "libpy/from_object.h"
#include "libpy/getattr.h"
#include "libpy/hash.h"
#include "libpy/to_object.h"
#include "test_utils.h"

namespace test_automethod {
using namespace std::literals;
using namespace py::cs::literals;

class automethod : public with_python_interpreter {
    std::forward_list<std::string> m_strings;
    std::forward_list<PyMethodDef> m_methods;

    void TearDown() override {
        m_strings.clear();
        m_methods.clear();
        with_python_interpreter::TearDown();
    }

public:
    template<auto f, int flags = 0>
    py::scoped_ref<> make_meth(const char* name, const char* doc = nullptr) {
        name = m_strings.emplace_front(name).data();
        if (doc) {
            doc = m_strings.emplace_front(doc).data();
        }

        auto& meth = m_methods.emplace_front(py::automethod<f, flags>(name, doc));
        py::scoped_ref out{PyCFunction_New(&meth, nullptr)};
        if (!out) {
            throw py::exception{};
        }
        return out;
    }

    template<auto f, int flags = 0>
    py::scoped_ref<> make_f(const char* name, const char* doc = nullptr) {
        name = m_strings.emplace_front(name).data();
        if (doc) {
            doc = m_strings.emplace_front(doc).data();
        }

        auto& meth = m_methods.emplace_front(py::autofunction<f, flags>(name, doc));
        py::scoped_ref out{PyCFunction_New(&meth, nullptr)};
        if (!out) {
            throw py::exception{};
        }
        return out;
    }
};

void test_f() {}
void test_meth(PyObject*) {}

TEST_F(automethod, name_no_doc) {
    std::string_view expected_name = "test_f"sv;

    auto test = [&](py::borrowed_ref<> f) {
        auto name = py::getattr_throws(f, "__name__");
        EXPECT_EQ(py::util::pystring_to_string_view(name), expected_name);

        auto doc = py::getattr_throws(f, "__doc__");
        EXPECT_EQ(doc.get(), Py_None);
    };

    test(make_f<test_f>(expected_name.data()));
    test(make_meth<test_meth>(expected_name.data()));
}

TEST_F(automethod, name_and_doc) {
    std::string_view expected_name = "test_f"sv;
    std::string_view expected_doc = "test_f's docstring"sv;

    auto test = [&](py::borrowed_ref<> f) {
        auto name = py::getattr_throws(f, "__name__");
        EXPECT_EQ(py::util::pystring_to_string_view(name), expected_name);

        auto doc = py::getattr_throws(f, "__doc__");
        EXPECT_EQ(py::util::pystring_to_string_view(doc), expected_doc);
    };

    test(make_f<test_f>(expected_name.data(), expected_doc.data()));
    test(make_meth<test_meth>(expected_name.data(), expected_doc.data()));
}

bool void_return_no_arg_called = false;
void void_return_no_arg() {
    void_return_no_arg_called = true;
}

TEST_F(automethod, void_return_no_arg) {
    auto f = make_f<void_return_no_arg>("f");

    ASSERT_FALSE(void_return_no_arg_called);
    auto res = py::call_function_throws(f);
    EXPECT_EQ(res.get(), Py_None);
    EXPECT_TRUE(void_return_no_arg_called);
}

PyObject* pyobject_return() {
    // return a list object so we don't need to worry about shared immutable references
    // that happen with ints or bytes.
    return py::to_object(std::vector<int>{1, 2, 3}).escape();
}

TEST_F(automethod, pyobject_return) {
    auto f = make_f<pyobject_return>("f");

    auto res = py::call_function_throws(f);
    // we are the sole owner of the reference created in our function
    EXPECT_EQ(Py_REFCNT(res.get()), 1);

    auto unboxed = py::from_object<std::vector<int>>(res);
    EXPECT_EQ(unboxed, (std::vector<int>{1, 2, 3}));
}

std::optional<std::tuple<int, float, bool>> void_return_fundamental_args_captured_call;
void void_return_fundamental_args(int a, float b, bool c) {
    void_return_fundamental_args_captured_call = std::make_tuple(a, b, c);
}

TEST_F(automethod, void_return_fundamental_args) {
    auto f = make_f<void_return_fundamental_args>("f");

    ASSERT_FALSE(void_return_fundamental_args_captured_call);
    auto res = py::call_function_throws(f, 1, 2.5, true);
    EXPECT_EQ(res.get(), Py_None);
    ASSERT_TRUE(void_return_fundamental_args_captured_call);
    EXPECT_EQ(*void_return_fundamental_args_captured_call, std::make_tuple(1, 2.5, true));
}

int fundamental_return_no_args() {
    return 12;
}

TEST_F(automethod, fundamental_return_no_args) {
    auto f = make_f<fundamental_return_no_args>("f");

    auto res = py::call_function_throws(f);
    int unboxed = py::from_object<int>(res);

    EXPECT_EQ(unboxed, 12);
}

std::tuple<std::size_t, std::size_t> string_view_argument(std::string_view view) {
    return {view.size(), py::hash_buffer(view.data(), view.size())};
}

TEST_F(automethod, string_view_argument) {
    auto f = make_f<string_view_argument>("f");

    std::string_view input_data = "ayy lmao"sv;
    std::size_t expected_hash = py::hash_buffer(input_data.data(), input_data.size());

    auto test = [&](py::borrowed_ref<> input) {
        auto res = py::call_function_throws(f, input);
        auto [size, hash] = py::from_object<std::tuple<std::size_t, std::size_t>>(res);
        EXPECT_EQ(size, input_data.size());
        EXPECT_EQ(hash, expected_hash);
    };

    py::scoped_ref bytes{PyBytes_FromString(input_data.data())};
    ASSERT_TRUE(bytes);
    test(bytes);

    py::scoped_ref byte_array{
        PyByteArray_FromStringAndSize(input_data.data(), input_data.size())};
    ASSERT_TRUE(byte_array);
    test(byte_array);

    py::scoped_ref immutible_view{PyMemoryView_FromObject(bytes.get())};
    ASSERT_TRUE(immutible_view);
    test(immutible_view);

    py::scoped_ref mutible_view{PyMemoryView_FromObject(byte_array.get())};
    ASSERT_TRUE(mutible_view);
    test(mutible_view);
}

std::tuple<std::size_t, std::size_t>
const_ref_string_view_argument(const std::string_view& view) {
    return string_view_argument(view);
}

TEST_F(automethod, const_ref_string_view_argument) {
    auto f = make_f<string_view_argument>("f");

    std::string_view input_data = "ayy lmao"sv;
    std::size_t expected_hash = py::hash_buffer(input_data.data(), input_data.size());

    // Use a byte array, which cannot be converted into a string_view through just the
    // fallback `from_object` handler; we must go through the buffer protocol handler.
    py::scoped_ref byte_array{
        PyByteArray_FromStringAndSize(input_data.data(), input_data.size())};
    ASSERT_TRUE(byte_array);

    auto res = py::call_function_throws(f, byte_array);
    auto [size, hash] = py::from_object<std::tuple<std::size_t, std::size_t>>(res);
    EXPECT_EQ(size, input_data.size());
    EXPECT_EQ(hash, expected_hash);
}

std::tuple<std::size_t, int> array_view(py::array_view<int> view) {
    return {view.size(), std::accumulate(view.begin(), view.end(), 0)};
}

TEST_F(automethod, array_view) {
    auto f = make_f<array_view>("f");

    std::vector<int> input_data = {1, 2, 3, 4, 5};
    int expected_sum = std::accumulate(input_data.begin(), input_data.end(), 0);

    auto test = [&](py::borrowed_ref<> input) {
        auto res = py::call_function_throws(f, input);
        auto [size, sum] = py::from_object<std::tuple<std::size_t, int>>(res);
        EXPECT_EQ(size, input_data.size());
        EXPECT_EQ(sum, expected_sum);
    };

    auto ndarray = py::move_to_numpy_array(std::vector<int>{input_data});
    ASSERT_TRUE(ndarray);
    test(ndarray);

    py::scoped_ref view{PyMemoryView_FromObject(ndarray.get())};
    ASSERT_TRUE(view);
    test(view);
}

TEST_F(automethod, array_view_wrong_ndim) {
    auto f = make_f<array_view>("f");

    std::array<std::size_t, 2> shape = {3ul, 5ul};
    std::array<std::ptrdiff_t, 2> strides = {5 * sizeof(int), sizeof(int)};
    // clang-format off
    std::vector<int> input_data = {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15
    };
    // clang-format on

    auto ndarray = py::move_to_numpy_array(std::vector<int>{input_data},
                                           py::new_dtype<int>(),
                                           shape,
                                           strides);
    ASSERT_TRUE(ndarray);

    auto res = py::call_function(f, ndarray);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "argument must be a 1 dimensional array, got ndim=2");
    PyErr_Clear();

    py::scoped_ref view{PyMemoryView_FromObject(ndarray.get())};
    ASSERT_TRUE(view);

    res = py::call_function(f, view);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "argument must be a 1 dimensional buffer, got ndim=2");
    PyErr_Clear();
}

TEST_F(automethod, array_view_wrong_dtype) {
    auto f = make_f<array_view>("f");

    auto ndarray = py::move_to_numpy_array(std::vector<float>{1.5f, 2.5f, 3.5f});
    ASSERT_TRUE(ndarray);

    auto res = py::call_function(f, ndarray);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(
        PyExc_TypeError, "expected array of dtype: int32, got array of type: float32");
    PyErr_Clear();

    py::scoped_ref view{PyMemoryView_FromObject(ndarray.get())};
    ASSERT_TRUE(view);

    res = py::call_function(f, view);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "cannot adapt buffer of format: "s +
                                      py::buffer_format<float> +
                                      " to an ndarray_view of type: int"s);
    PyErr_Clear();
}

template<typename unit, typename I>
py::datetime64<unit> sum_datetime64_int_values(I begin, I end) {
    return py::datetime64<unit>{
        std::accumulate(begin, end, 0, [](std::int64_t a, py::datetime64<unit> b) {
            return a + static_cast<std::int64_t>(b);
        })};
}

std::tuple<std::size_t, py::datetime64ns>
datetime64_array_view(py::array_view<py::datetime64ns> view) {
    return {view.size(),
            sum_datetime64_int_values<py::chrono::ns>(view.begin(), view.end())};
}

TEST_F(automethod, datetime64_array_view) {
    auto f = make_f<datetime64_array_view>("f");

    std::vector<py::datetime64ns> input_data = {py::datetime64ns{1},
                                                py::datetime64ns{2},
                                                py::datetime64ns{3}};
    std::size_t expected_size = input_data.size();
    py::datetime64ns expected_sum =
        sum_datetime64_int_values<py::chrono::ns>(input_data.begin(), input_data.end());

    auto ndarray = py::move_to_numpy_array(std::move(input_data));
    ASSERT_TRUE(ndarray);

    auto res = py::call_function_throws(f, ndarray);
    auto [size, sum] = py::from_object<std::tuple<std::size_t, py::datetime64ns>>(res);
    EXPECT_EQ(size, expected_size);
    EXPECT_EQ(sum, expected_sum);
}

std::tuple<std::array<std::size_t, 2>, std::vector<int>>
two_dimensional_array_view(py::ndarray_view<int, 2> view) {
    std::vector<int> sums(view.shape()[0]);
    for (std::size_t row = 0; row < view.shape()[0]; ++row) {
        for (std::size_t col = 0; col < view.shape()[1]; ++col) {
            sums[row] += view(row, col);
        }
    }
    return {view.shape(), sums};
}

TEST_F(automethod, two_dimensional_array_view) {
    auto f = make_f<two_dimensional_array_view>("f");

    std::array<std::size_t, 2> shape = {3ul, 5ul};
    std::array<std::ptrdiff_t, 2> strides = {5 * sizeof(int), sizeof(int)};
    // clang-format off
    std::vector<int> input_data = {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15
    };
    // clang-format on
    std::vector<int> expected_sums = {
        1 + 2 + 3 + 4 + 5,
        6 + 7 + 8 + 9 + 10,
        11 + 12 + 13 + 14 + 15,
    };

    auto test = [&](py::borrowed_ref<> input) {
        auto res = py::call_function_throws(f, input);
        auto [shape, sums] =
            py::from_object<std::tuple<std::array<std::size_t, 2>, std::vector<int>>>(
                res);
        EXPECT_EQ(shape, shape);
        EXPECT_EQ(sums, expected_sums);
    };

    auto ndarray = py::move_to_numpy_array(std::vector<int>{input_data},
                                           py::new_dtype<int>(),
                                           shape,
                                           strides);
    ASSERT_TRUE(ndarray);
    test(ndarray);

    py::scoped_ref view{PyMemoryView_FromObject(ndarray.get())};
    ASSERT_TRUE(view);
    test(view);
}

TEST_F(automethod, two_dimensional_array_view_wrong_ndim) {
    auto f = make_f<two_dimensional_array_view>("f");

    auto ndarray = py::move_to_numpy_array(std::vector<int>{1, 2, 3, 4});
    ASSERT_TRUE(ndarray);

    auto res = py::call_function(f, ndarray);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "argument must be a 2 dimensional array, got ndim=1");
    PyErr_Clear();

    py::scoped_ref view{PyMemoryView_FromObject(ndarray.get())};
    ASSERT_TRUE(view);

    res = py::call_function(f, view);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "argument must be a 2 dimensional buffer, got ndim=1");
    PyErr_Clear();
}

std::tuple<std::string, std::size_t, std::size_t>
mut_any_ndarray_view(py::array_view<py::any_ref> view) {
    if (static_cast<std::size_t>(view.strides()[0]) != view.vtable().size()) {
        throw py::exception(PyExc_AssertionError, "input should not be strided");
    }

    return {view.vtable().type_name(),
            view.size(),
            py::hash_buffer(reinterpret_cast<const char*>(view.buffer()),
                            view.size() * view.vtable().size())};
}

TEST_F(automethod, mut_any_ndarray_view) {
    auto f = make_f<mut_any_ndarray_view>("f");

    auto test = [&](auto... data) {
        std::vector vec = {data...};
        using T = typename decltype(vec)::value_type;
        std::string expected_type_name = py::util::type_name<T>();
        std::size_t expected_size = vec.size();
        std::size_t expected_hash = py::hash_buffer(reinterpret_cast<const char*>(
                                                        vec.data()),
                                                    vec.size() * sizeof(T));

        auto ndarray = py::move_to_numpy_array(std::move(vec));
        ASSERT_TRUE(ndarray);

        auto res = py::call_function_throws(f, ndarray);
        auto [type_name, size, hash] =
            py::from_object<std::tuple<std::string, std::size_t, std::size_t>>(res);

        EXPECT_EQ(type_name, expected_type_name);
        EXPECT_EQ(size, expected_size);
        EXPECT_EQ(hash, expected_hash);

        if constexpr (!std::is_same_v<T, py::datetime64ns>) {
            // cannot take memoryview over datetime dtype arrays
            py::scoped_ref view{PyMemoryView_FromObject(ndarray.get())};
            ASSERT_TRUE(view);
            res = py::call_function_throws(f, view);
            auto [type_name, size, hash] =
                py::from_object<std::tuple<std::string, std::size_t, std::size_t>>(res);

            EXPECT_EQ(type_name, expected_type_name);
            EXPECT_EQ(size, expected_size);
            EXPECT_EQ(hash, expected_hash);
        }
    };

    test(1, 2, 3, 4);
    test(1L, 2L, 3L, 4L);
    test(1u, 2u, 3u, 4u);
    test(1.5f, 2.5f, 3.5f, 4.5f);
    test(1.5, 2.5, 3.5, 4.5);
    test(py::datetime64ns{1},
         py::datetime64ns{2},
         py::datetime64ns{3},
         py::datetime64ns{4});
}

std::tuple<std::string, std::size_t, std::size_t>
immut_any_ndarray_view(py::array_view<py::any_cref> view) {
    if (static_cast<std::size_t>(view.strides()[0]) != view.vtable().size()) {
        throw py::exception(PyExc_AssertionError, "input should not be strided");
    }

    return {view.vtable().type_name(),
            view.size(),
            py::hash_buffer(reinterpret_cast<const char*>(view.buffer()),
                            view.size() * view.vtable().size())};
}

TEST_F(automethod, immut_any_ndarray_view) {
    auto f = make_f<immut_any_ndarray_view>("f");

    auto test = [&](auto... data) {
        std::vector vec = {data...};
        using T = typename decltype(vec)::value_type;
        std::string expected_type_name = py::util::type_name<T>();
        std::size_t expected_size = vec.size();
        std::size_t expected_hash = py::hash_buffer(reinterpret_cast<const char*>(
                                                        vec.data()),
                                                    vec.size() * sizeof(T));

        auto ndarray = py::move_to_numpy_array(std::move(vec));
        ASSERT_TRUE(ndarray);

        auto res = py::call_function_throws(f, ndarray);
        auto [type_name, size, hash] =
            py::from_object<std::tuple<std::string, std::size_t, std::size_t>>(res);

        EXPECT_EQ(type_name, expected_type_name);
        EXPECT_EQ(size, expected_size);
        EXPECT_EQ(hash, expected_hash);

        if constexpr (!std::is_same_v<T, py::datetime64ns>) {
            // cannot take memoryview over datetime dtype arrays
            py::scoped_ref view{PyMemoryView_FromObject(ndarray.get())};
            ASSERT_TRUE(view);
            res = py::call_function_throws(f, view);
            auto [type_name, size, hash] =
                py::from_object<std::tuple<std::string, std::size_t, std::size_t>>(res);

            EXPECT_EQ(type_name, expected_type_name);
            EXPECT_EQ(size, expected_size);
            EXPECT_EQ(hash, expected_hash);
        }
    };

    test(1, 2, 3, 4);
    test(1L, 2L, 3L, 4L);
    test(1u, 2u, 3u, 4u);
    test(1.5f, 2.5f, 3.5f, 4.5f);
    test(1.5, 2.5, 3.5, 4.5);
    test(py::datetime64ns{1},
         py::datetime64ns{2},
         py::datetime64ns{3},
         py::datetime64ns{4});
}

int optional_arg(py::arg::opt<int> a) {
    return a.get().value_or(-1);
}

TEST_F(automethod, optional_arg) {
    auto f = make_f<optional_arg>("f");

    auto res = py::call_function_throws(f);
    int unboxed = py::from_object<int>(res);
    EXPECT_EQ(unboxed, -1);

    res = py::call_function_throws(f, 12);
    unboxed = py::from_object<int>(res);
    EXPECT_EQ(unboxed, 12);
}

int req_then_optional_arg(int a, py::arg::opt<int> b) {
    return a + b.get().value_or(0);
}

TEST_F(automethod, req_then_optional_arg_missing) {
    auto f = make_f<req_then_optional_arg>("f");

    auto res = py::call_function(f);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(
        PyExc_TypeError, "function takes from 1 to 2 arguments but 0 were given");
    PyErr_Clear();
}

TEST_F(automethod, req_then_optional_arg_too_many) {
    auto f = make_f<req_then_optional_arg>("f");

    auto res = py::call_function(f, 1, 2, 3);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(
        PyExc_TypeError, "function takes from 1 to 2 arguments but 3 were given");
    PyErr_Clear();
}

int no_keywords(int a) {
    return a;
}

TEST_F(automethod, no_keywords) {
    auto f = make_f<no_keywords>("f");

    auto res = EVAL_PYTHON("f(a=12)", {{"f", f}});
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError, "function takes no keyword arguments");
    PyErr_Clear();
}

TEST_F(automethod, not_enough_positional_only_1) {
    auto f = make_f<no_keywords>("f");

    auto res = py::call_function(f);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "function takes 1 argument but 0 were given");
    PyErr_Clear();
}

TEST_F(automethod, too_many_positional_only_1) {
    auto f = make_f<no_keywords>("f");

    auto res = py::call_function(f, 1, 2);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "function takes 1 argument but 2 were given");
    PyErr_Clear();
}

int three_positional(int a, int b, int c) {
    return a + b + c;
}

TEST_F(automethod, not_enough_positional) {
    auto f = make_f<three_positional>("f");

    auto res = py::call_function(f, 1, 2);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "function takes 3 arguments but 2 were given");
    PyErr_Clear();
}

TEST_F(automethod, too_many_positional) {
    auto f = make_f<three_positional>("f");

    auto res = py::call_function(f, 1, 2, 3, 4);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "function takes 3 arguments but 4 were given");
    PyErr_Clear();
}

int keyword_arg(py::arg::kwd<decltype("kw_arg"_cs), int> kw_arg) {
    return kw_arg.get();
}

TEST_F(automethod, keyword_arg) {
    auto f = make_f<keyword_arg>("f");

    auto res = py::call_function(f);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "function takes 1 argument but 0 were given");
    PyErr_Clear();

    // call positionally
    res = py::call_function_throws(f, 12);
    int unboxed = py::from_object<int>(res);
    EXPECT_EQ(unboxed, 12);

    res = EVAL_PYTHON("f(kw_arg=12)", {{"f", f}});
    ASSERT_TRUE(res);
    unboxed = py::from_object<int>(res);
    EXPECT_EQ(unboxed, 12);
}

int opt_keyword_arg(py::arg::opt_kwd<decltype("kw_arg"_cs), int> kw_arg) {
    return kw_arg.get().value_or(-1);
}

TEST_F(automethod, opt_keyword_arg) {
    auto f = make_f<opt_keyword_arg>("f");

    auto res = py::call_function_throws(f);
    int unboxed = py::from_object<int>(res);
    EXPECT_EQ(unboxed, -1);

    // call positionally
    res = py::call_function_throws(f, 12);
    unboxed = py::from_object<int>(res);
    EXPECT_EQ(unboxed, 12);

    res = EVAL_PYTHON("f(kw_arg=12)", {{"f", f}});
    ASSERT_TRUE(res);
    unboxed = py::from_object<int>(res);
    EXPECT_EQ(unboxed, 12);
}

struct non_copyable_type {
    non_copyable_type() = default;
    non_copyable_type(const non_copyable_type&) = delete;
    non_copyable_type(non_copyable_type&&) = default;
};
}  // namespace test_automethod

namespace py::dispatch {
template<>
struct from_object<test_automethod::non_copyable_type> {
    static test_automethod::non_copyable_type f(py::borrowed_ref<>) {
        return {};
    }
};
}  // namespace py::dispatch

namespace test_automethod {
// Take `non_copyable_type` by _value_ to test that we are properly forwarding values all
// the way down to the function. This test is just to ensure we can actually compile, it
// doesn't need to do much at runtime.
void non_copyable(non_copyable_type) {}

TEST_F(automethod, non_copyable) {
    auto f = make_f<non_copyable>("f");

    auto res = py::call_function_throws(f, true);
    EXPECT_EQ(res.get(), Py_None);
}

double double_mut_ref(double& a) {
    return a;
}

TEST_F(automethod, int_mut_ref) {
    auto f = make_f<double_mut_ref>("f");

    auto res = py::call_function(f, 12.0);
    EXPECT_FALSE(res);
    expect_pyerr_type_and_message(
        PyExc_TypeError,
        "failed to convert Python object of type float to a C++ "
        "object of type double&: ob=12.0");
    PyErr_Clear();
}

int double_const_ref(const double& a) {
    return a;
}

TEST_F(automethod, int_const_ref) {
    auto f = make_f<double_const_ref>("f");

    auto res = py::call_function_throws(f, 12.0);
    double unboxed = py::from_object<int>(res);
    EXPECT_EQ(unboxed, 12.0);
}
}  // namespace test_automethod
