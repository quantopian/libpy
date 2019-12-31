#include <string>

#ifndef LIBPY_VALGRIND
#define CALLGRIND_ZERO_STATS
#define CALLGRIND_START_INSTRUMENTATION
#define CALLGRIND_STOP_INSTRUMENTATION
#define CALLGRIND_DUMP_STATS_AT(x)
#else
#include <valgrind/callgrind.h>
#endif

#include <iostream>

namespace py::valgrind {

#ifndef LIBPY_VALGRIND
constexpr bool enabled = false;
#else
constexpr bool enabled = true;
#endif

/** An RAII object for running callgrind for a given section.

    This is a nop by default, to actually instrument the section: compile with
    ``-LIBPY_VALGRIND`` and run python under valgrind like:

    \code
    $ valgrind --tool=callgrind \
        --instr-atstart=no \
        --dump-line=yes \
        --dump-instr=yes \
        --cache-sim=yes \
        --branch-sim=yes \
        python
    \endcode

    The `--instr-atstart=no` says to not start collecting right away and to
    wait for our RAII object to trigger collection.

    ### Notes
    This is not reentrant.
 */
struct callgrind final {
private:
    std::string m_tag;

public:
    /** Collect stats for the given scope.

        @param tag The tag for the output stats.
     */
    inline callgrind(const std::string& tag) : m_tag(tag) {
        if (enabled) {
            CALLGRIND_ZERO_STATS;
            CALLGRIND_START_INSTRUMENTATION;
        }
    }

    inline ~callgrind() {
        if (enabled) {
            CALLGRIND_STOP_INSTRUMENTATION;
            CALLGRIND_DUMP_STATS_AT(m_tag.c_str());
        }
    }
};
}  // namespace py::valgrind
