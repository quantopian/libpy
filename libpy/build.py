import os
import sys

import setuptools
import numpy as np


def get_include():
    """Get the path to the libpy headers relative to the installed package.

    Returns
    -------
    include_path : str
        The path to the libpy include directory.
    """
    return os.path.join(os.path.dirname(__file__), 'include')


class LibpyExtension(setuptools.Extension):
    """A :class:`setuptools.Extension` replacement for libpy based extensions.

    Parameters
    ----------
    *args
        All positional arguments forwarded to :class:`setuptools.Extension`.
    optlevel : int, optional
        The optimization level to forward to the C++ compiler.
        Defaults to 0.
    debug_symbols : bool, optional
        Should debug symbols be generated?
        Defaults to True.
    use_libpy_suggested_warnings : bool, optional
        Should libpy add it's default set of warnings to the compiler flags.
        This set is picked to aid code clarity and attempt to common mistakes.
        Defaults to True.
    werror : bool, optional
        Treat warnings as errors. The libpy developers believe that most
        compiler warnings indicate serious problems and should fail the build.
        Defaults to True.
    max_errors : int or None, optional
        Limit the number of error messages that are shown.
        Defaults to None, showing all error messages.
    **kwargs
        All other keyword arguments forwarded to :class:`setuptools.Extension`.

    Notes
    -----
    This class sets the `language` field to `c++` because libpy only works with
    C++.

    This class also passes `-std=gnu++17` which is the minimum language
    standard required by libpy.

    Any compiler flags added by libpy will appear *before*
    `extra_compile_args`. This gives the user the ability to override any of
    libpy's options.
    """
    def __init__(self, *args, **kwargs):
        kwargs['language'] = 'c++'

        libpy_extra_compile_args = [
            '-std=gnu++17',
            '-pipe',
            '-DPY_MAJOR_VERSION=%d' % sys.version_info.major,
            '-DPY_MINOR_VERSION=%d' % sys.version_info.minor,
        ]

        optlevel = kwargs.pop('optlevel', 0)
        libpy_extra_compile_args.append('-O%d' % optlevel)
        if kwargs.pop('debug_symbols', True):
            libpy_extra_compile_args.append('-g')

        if kwargs.pop('use_libpy_suggested_warnings', True):
            libpy_extra_compile_args.extend([
                '-Wall',
                '-Wextra',
                '-Wno-register',
                '-Wno-missing-field-initializers',
                '-Wsign-compare',
                '-Wsuggest-override',
                '-Wparentheses',
                '-Waggressive-loop-optimizations',
            ])
        if kwargs.pop('werror', True):
            libpy_extra_compile_args.append('-Werror')
        max_errors = kwargs.pop('max_errors', None)
        if max_errors is not None:
            libpy_extra_compile_args.append('-fmax-errors=%d' % max_errors)
        kwargs['extra_compile_args'] = (
            libpy_extra_compile_args +
            kwargs.setdefault('extra_compile_args', [])
        )

        include_dirs = kwargs.setdefault('include_dirs', [])
        include_dirs.append(get_include())
        include_dirs.append(np.get_include())

        super(LibpyExtension, self).__init__(*args, **kwargs)
