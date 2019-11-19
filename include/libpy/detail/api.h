#pragma once

/** Marker for a single function or type to indicate that it should be exported in the
    shared object.

    # Examples
    ```
    LIBPY_EXPORT int public_function(float);

    struct LIBPY_EXPORT public_struct {};
    ```

    @note The visibility modifiers for types must appear between the initial keyword and
    the name of the type. Type visibility is applied only to "vague linkage entities"
    associated with the type. For example a vtable or typeinfo node. Public types do not
    automatically make all of their members public.
 */
#define LIBPY_EXPORT __attribute__((visibility("default")))

/** Marker for a single function or type to indicate that it should not be exported in the
    libpy shared object. The default visibility is *hidden*, so this can be used inside a
    `LIBPY_BEGIN_EXPORT/LIBPY_END_EXPORT` block to turn off only some types and functions.

    # Examples
    ```
    LIBPY_NO_EXPORT int hidden_function(float);

    struct LIBPY_NO_EXPORT hidden_struct {};
    ```

    @note The visibility modifiers for types must appear between the initial keyword and
    the name of the type. Type visibility is applied only to "vague linkage entities"
    associated with the type. For example a vtable or typeinfo node. Public types do not
    automatically make all of their members public.
 */
#define LIBPY_NO_EXPORT __attribute__((visibility("hidden")))

/** Temporarily change the default visibility to public.

    # Examples
    ```
    int hidden_function(float);

    LIBPY_BEGIN_EXPORT
    int public_function(float);
    LIBPY_END_EXPORT

    int another_hidden_function(float);
    ```

    @see LIBPY_END_EXPORT
 */
#define LIBPY_BEGIN_EXPORT _Pragma("GCC visibility push(default)")

/** Close the scope entered by `LIBPY_BEGIN_EXPORT`.

    @see LIBPY_BEGIN_EXPORT
 */
#define LIBPY_END_EXPORT _Pragma("GCC visibility pop")
