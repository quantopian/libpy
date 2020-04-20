#include <tuple>
#include <type_traits>
#include <utility>

#include "libpy/meta.h"

namespace py::detail {
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

    \code
    using my_table = py::table<py::C<int>("some_name"_cs),
                               py::C<double>("some_other_name"_cs)>;
    \endcode
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

template<typename C, typename Mappings, typename = void>
struct relabeled_column_name_impl {
    using type = C;
};

template<typename C, typename... From, typename... To>
struct relabeled_column_name_impl<
    C,
    std::tuple<std::pair<From, To>...>,
    std::enable_if_t<py::meta::element_of<C, std::tuple<From...>>>> {

    using type = std::tuple_element_t<py::meta::search_tuple<C, std::tuple<From...>>,
                                      std::tuple<To...>>;
};

/** Given a column name, and a set of relabel mappings to apply, get the new column
    name. This is used to help implement `relabel()` on `row_view` and `table_view`.

    @tparam C The name of the column to lookup.
    @tparam Mappings A `std::tuple` of `std::pair`s mapping old column names to new column
            names. If `C` is the value of `first_type` on any of the pairs, the result
            will be that same pair's `second_type`. Otherwise, `C` will be returned
            unchanged.
 */
template<typename C, typename Mappings>
using relabeled_column_name = typename relabeled_column_name_impl<C, Mappings>::type;
}  // namespace py::detail
