import glob
import os
import subprocess
import sys
import warnings

import setuptools
import numpy as np

import libpy


def detect_compiler():
    p = subprocess.Popen(
        [
            os.path.join(os.path.dirname(__file__), '_build-and-run'),
            os.path.join(os.path.dirname(__file__), '_detect-compiler.cc'),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    stdout, _ = p.communicate()
    if stdout == b'GCC\n':
        return 'GCC'
    elif stdout == b'CLANG\n':
        return 'CLANG'

    warnings.warn(
        'Could not detect which compiler is being used, assuming gcc.',
    )
    return 'GCC'


def get_include():
    """Get the path to the libpy headers relative to the installed package.

    Returns
    -------
    include_path : str
        The path to the libpy include directory.
    """
    return os.path.join(os.path.dirname(__file__), 'include')


class LibpyExtension(setuptools.Extension, object):
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
    ubsan : bool, optional
        Compile with ubsan? Implies optlevel=0.
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
    _compiler = detect_compiler()

    _recommended_warnings = [
        '-Werror',
        '-Wall',
        '-Wextra',
        '-Wno-register',
        '-Wno-missing-field-initializers',
        '-Wsign-compare',
        '-Wparentheses',
    ]
    if _compiler == 'GCC':
        _recommended_warnings.extend([
            '-Wsuggest-override',
            '-Wno-maybe-uninitialized',
            '-Waggressive-loop-optimizations',
        ])
    elif _compiler == 'CLANG':
        _recommended_warnings.extend([
            '-Wno-gnu-string-literal-operator-template',
            '-Wno-missing-braces',
            '-Wno-self-assign-overloaded',
        ])
    else:
        raise AssertionError('unknown compiler: %s' % _compiler)

    _base_flags = [
        '-std=gnu++17',
        '-pipe',
        '-fvisibility-inlines-hidden',
        '-DPY_MAJOR_VERSION=%d' % sys.version_info.major,
        '-DPY_MINOR_VERSION=%d' % sys.version_info.minor,
        '-DLIBPY_MAJOR_VERSION=%d' % libpy.version_info.major,
        '-DLIBPY_MINOR_VERSION=%d' % libpy.version_info.minor,
        '-DLIBPY_MICRO_VERSION=%d' % libpy.version_info.micro,
    ]

    def __init__(self, *args, **kwargs):
        kwargs['language'] = 'c++'

        libpy_extra_compile_args = self._base_flags.copy()

        libpy_extra_link_args = []

        optlevel = kwargs.pop('optlevel', 0)
        ubsan = kwargs.pop('ubsan', False)
        if ubsan:
            optlevel = 0
            libpy_extra_compile_args.append('-fsanitize=undefined')
            libpy_extra_link_args.append('-lubsan')

        libpy_extra_compile_args.append('-O%d' % optlevel)
        if kwargs.pop('debug_symbols', True):
            libpy_extra_compile_args.append('-g')

        if kwargs.pop('use_libpy_suggested_warnings', True):
            libpy_extra_compile_args.extend(self._recommended_warnings)

        if kwargs.pop('werror', True):
            libpy_extra_compile_args.append('-Werror')

        max_errors = kwargs.pop('max_errors', None)
        if max_errors is not None:
            if self._compiler == 'GCC':
                libpy_extra_compile_args.append(
                    '-fmax-errors=%d' % max_errors,
                )
            elif self._compiler == 'CLANG':
                libpy_extra_compile_args.append(
                    '-ferror-limit=%d' % max_errors,
                )
            else:
                raise AssertionError('unknown compiler: %s' % self._compiler)

        kwargs['extra_compile_args'] = (
            libpy_extra_compile_args +
            kwargs.get('extra_compile_args', [])
        )
        kwargs['extra_link_args'] = (
            libpy_extra_link_args +
            kwargs.get('extra_link_args', [])
        )
        depends = kwargs.setdefault('depends', []).copy()
        depends.extend(glob.glob(get_include() + '/**/*.h', recursive=True))
        depends.extend(glob.glob(np.get_include() + '/**/*.h', recursive=True))

        include_dirs = kwargs.setdefault('include_dirs', [])
        include_dirs.append(get_include())
        include_dirs.append(np.get_include())

        super(LibpyExtension, self).__init__(*args, **kwargs)
