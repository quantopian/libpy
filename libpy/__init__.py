import ctypes
import os
from .version import __version__  # noqa


_so = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), 'libpy.so'),
    ctypes.RTLD_GLOBAL,
)


class VersionInfo(ctypes.Structure):
    _fields_ = [
        ('major', ctypes.c_int),
        ('minor', ctypes.c_int),
        ('patch', ctypes.c_int),
    ]

    def __repr__(self):
        return (
            '{type_name}(major={0.major},'
            ' minor={0.minor}, patch={0.patch})'
        ).format(self, type_name=type(self).__name__)

    def __str__(self):
        return '{0.major}.{0.minor}.{0.patch}'.format(self)


version_info = VersionInfo.in_dll(_so, 'libpy_abi_version')
__abi_version__ = str(version_info)
