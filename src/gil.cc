#include "libpy/gil.h"

namespace py {
thread_local PyThreadState* gil::m_save;
}  // namespace py
