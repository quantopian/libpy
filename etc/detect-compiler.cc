#include <iostream>

int main() {
    const char* name;
#if defined(__clang__)
    name = "CLANG";
#elif defined (__GNUC__)
    name = "GCC";
#else
    name = "UNKNOWN";
#endif
    std::cout << name << '\n';
    return 0;
}
