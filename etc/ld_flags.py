# via https://github.com/python/cpython/blob/deb016224cc506503fb05e821a60158c83918ed4/Misc/python-config.in#L50 # noqa

import sysconfig

libs = []
libpl = sysconfig.get_config_vars('LIBPL')
if libpl:
    libs.append("-L"+libpl[0])

libpython = sysconfig.get_config_var('LIBPYTHON')
if libpython:
    libs.append(libpython)
libs.extend(sysconfig.get_config_vars("LIBS", "SYSLIBS"))
print(' '.join(libs))
