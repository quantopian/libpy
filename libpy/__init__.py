import ctypes
import os


_so = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), 'libpy.so'),
    ctypes.RTLD_GLOBAL,
)


class _version_info(ctypes.Structure):
    _fields_ = [
        ('major', ctypes.c_int),
        ('minor', ctypes.c_int),
        ('patch', ctypes.c_int),
    ]

    def __repr__(self):
        return (
            'libpy._version_info(major={0.major},'
            ' minor={0.minor}, patch={0.patch})'
        ).format(self)

    def __str__(self):
        return '{0.major}.{0.minor}.{0.patch}'.format(self)


version_info = _version_info.in_dll(_so, 'libpy_abi_version')
__version__ = str(version_info)
