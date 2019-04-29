#pragma once

#include <utility>

namespace py::util {
template<typename F>
struct scope_guard {
private:
    F m_callback;

public:
    scope_guard(F&& callback) : m_callback(std::move(callback)) {}

    ~scope_guard() {
        m_callback();
    }
};
}  // namespace py::util
