#include <string>
#include <vector>

#include <libpy/abi.h>
#include <libpy/autoclass.h>
#include <libpy/exception.h>

namespace libpy_tutorial {

class sample_class {
private:
    int m_a;
    float m_b;

public:
    sample_class(int a, float b) : m_a(a), m_b(b) {}

    int a() const {
        return m_a;
    }

    float b() const {
        return m_b;
    }

    float sum() const {
        return m_a + m_b;
    }

    float sum_plus(float arg) const {
        return sum() + arg;
    }

    double operator()(int a, double b) const {
        return m_b + a + b;
    }

    int operator+(const sample_class& other) const {
        return m_a + other.a();
    }
    bool operator>(const sample_class& other) const {
        return m_a > other.a();
    }
    int operator-() {
        return -m_a;
    }
    explicit operator std::int64_t() const {
        return m_a;
    }
};

namespace {

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "libpy_tutorial.classes",
    nullptr,
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit_classes() {
    if (py::abi::ensure_compatible_libpy_abi()) {
        return nullptr;
    }
    import_array();
    auto m = py::owned_ref(PyModule_Create(&module));
    if (!m) {
        return nullptr;
    }
    try {
        auto type = py::autoclass<sample_class>("SampleClass")
                        .new_<int, float>()                   //__new__ takes parameters
                        .doc("Small docstring for my class")  // add a class docstring
                        .def<&sample_class::a>("a")
                        .def<&sample_class::b>("b")
                        .def<&sample_class::sum>("sum")
                        .def<&sample_class::sum_plus>("sum_plus")
                        .callable<int, double>()
                        .arithmetic<sample_class, int>()    // define artithmetic
                        .comparisons<sample_class, bool>()  // define comparisons
                        .unary()                            // unary ops
                        .conversions()                      // type conversions
                        //.repr<&sample_class::repr>()
                        //.hash()
                        //.iter()
                        // .string()
                        // .hash()
                        .type();
        if (PyObject_SetAttrString(m.get(),
                                   "SampleClass",
                                   static_cast<PyObject*>(type))) {
            return nullptr;
        }
    }
    catch (const std::exception& e) {
        return py::raise_from_cxx_exception(e);
    }

    return std::move(m).escape();
}
}  // namespace
}  // namespace libpy_tutorial
