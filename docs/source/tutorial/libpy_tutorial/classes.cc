#include <cmath>
#include <string>
#include <vector>

#include <libpy/autoclass.h>
#include <libpy/automodule.h>
#include <libpy/exception.h>

namespace libpy_tutorial {
class vec3d {
private:
    std::array<double, 3> m_values;

public:
    vec3d(double x, double y, double z) : m_values({x, y, z}) {}

    double x() const {
        return m_values[0];
    }

    double y() const {
        return m_values[1];
    }

    double z() const {
        return m_values[2];
    }

    vec3d operator+(const vec3d& other) const {
        return {x() + other.x(), y() + other.y(), z() + other.z()};
    }

    vec3d operator-(const vec3d& other) const {
        return {x() - other.x(), y() - other.y(), z() - other.z()};
    }

    double operator*(const vec3d& other) const {
        return std::inner_product(m_values.begin(),
                                  m_values.end(),
                                  other.m_values.begin(),
                                  0.0);
    }

    double magnitude() const {
        return std::sqrt(*this * *this);
    }
};

std::ostream& operator<<(std::ostream& s, const vec3d& v) {
    return s << '{' << v.x() << ", " << v.y() << ", " << v.z() << '}';
}

// `repr` could also be a member function, but free functions are useful for adding
// a Python repr without modifying the methods of the type.
std::string repr(const vec3d& v) {
    std::stringstream ss;
    ss << "Vec3d(" << v.x() << ", " << v.y() << ", " << v.z() << ')';
    return ss.str();
}
}  // namespace libpy_tutorial

namespace py::dispatch {
// Make it possible to convert a `vec3d` into a Python object.
template<>
struct LIBPY_NO_EXPORT to_object<libpy_tutorial::vec3d>
    : public py::autoclass<libpy_tutorial::vec3d>::to_object {};
}  // namespace py::dispatch

namespace libpy_tutorial {

LIBPY_AUTOMODULE(libpy_tutorial, classes, ({}))
(py::borrowed_ref<> m) {
    py::owned_ref t =
        py::autoclass<vec3d>(PyModule_GetName(m.get()) + std::string(".Vec3d"))
            .doc("An efficient 3-vector.")   // add a class docstring
            .new_<double, double, double>()  //__new__ takes parameters
            // bind the named methods to Python
            .def<&vec3d::x>("x")
            .def<&vec3d::y>("y")
            .def<&vec3d::z>("z")
            .def<&vec3d::magnitude>("magnitude")
            .str()         // set `operator<<(std::ostream&, vec3d) to `str(x)` in Python
            .repr<repr>()  // set `repr` to be the result of `repr(x)` in Python
            .arithmetic<vec3d>()  // bind the arithmetic operators to their Python
                                  // equivalents
            .type();
    return PyObject_SetAttrString(m.get(), "Vec3d", static_cast<PyObject*>(t));
}
}  // namespace libpy_tutorial
