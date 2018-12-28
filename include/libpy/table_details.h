#include <tuple>
#include <utility>

namespace py {
namespace detail {

/** Sentinel struct used to store a name and a type for py::table and related types.

    The classes defined in table.h (table, table_view, row, and row_view) are all
    templated on a variable number of "column singletons", which are pointers to
    specializations of ``column``.

    We use pointers because we want to be able to template tables on "columns",
    which carry two pieces of information:

    1. A name (e.g. "asof_date").
    2. A type (e.g. "datetime64").

    Templating on types is straightforward, but templating on names is hard to make
    ergonomic. We can use py::char_sequence literals ("foo"_cs) to define values that
    encode compile-time strings, but we can't make them template parameters directly
    without requiring heavy use of ``decltype`` for clients of ``table``, because
    char_sequence values that can't be used as template parameters.

    What we can do, however, is make ``py::C`` be a function that returns a **pointer** to
    an instance of the type we want to encode, and to template ``table`` and friends on
    those pointers. Inside the table and row types, we can use `py::unwrap_column` on the
    value to get the column type that it is a pointer to.

    All of this enables the following, relatively pleasant syntax for consumers:

    ```
    using my_table = py::table<py::C<int>("some_name"_cs),
                               py::C<double>("some_other_name"_cs)>;
    ```
*/
template<typename Key, typename Value>
struct column {
    using key = Key;
    using value = Value;

    using const_column = column<Key, std::add_const_t<Value>>;
    using remove_const_column = column<Key, std::remove_const_t<Value>>;
};

/** Variable template for sentinel instances of ``column``.

    We use addresses of these values as template parameters for ``table`` and its
    associated types.
*/
template<typename T>
T column_singleton;

/** Helper for unwrapping a column singleton to get the underlying column type.
 */
template<auto p>
using unwrap_column = typename std::remove_pointer_t<decltype(p)>;

/** Convert a column with ``T`` values to a one with ``const T`` values.
 */
template<auto p, typename C = typename unwrap_column<p>::const_column>
constexpr C* const_column = &column_singleton<C>;

/** Convert a column with ``const T`` values to a one with ``T`` values.
 */
template<auto p, typename C = typename unwrap_column<p>::remove_const_column>
constexpr C* remove_const_column = &column_singleton<C>;

}  // namespace detail

}  // namespace py
