#pragma once

#include <chrono>
#include <iostream>

namespace py::util {
class timer {
private:
    using clock = std::chrono::high_resolution_clock;
    const char* m_msg;
    clock::time_point m_start;

public:
    inline timer(const char* msg = nullptr) : m_msg(msg), m_start(clock::now()) {}

    inline ~timer() {
        clock::time_point end = clock::now();
        if (m_msg) {
            std::cout << m_msg << ": ";
        }
        std::cout << (end - m_start).count() << " nanoseconds\n";
    }
};
}
