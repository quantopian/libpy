import ast
from distutils.command.build_py import build_py as _build_py
import os
import pathlib
import shutil
import stat

from setuptools import setup


class BuildFailed(Exception):
    pass


# Setting ``LIBPY_DONT_BUILD`` to a truthy value will disable building the
# libpy extension, but allow the setup.py to run so that the Python support
# code may be installed. This exists to allow libpy to be used with alternative
# which may produce the libpy shared object in an alternative way. This flag
# prevents the setup.py from attempting to rebuild the shared object which may
# clobber or duplicate work done by the larger build system. This is an
# advanced feature and shouldn't be used without care as it may produce invalid
# installs of ``libpy``.
dont_build = ast.literal_eval(os.environ.get('LIBPY_DONT_BUILD', '0'))


class build_py(_build_py):
    def run(self):
        if self.dry_run or dont_build:
            return super().run()

        super().run()
        path = os.path.dirname(os.path.abspath(__file__))
        command = 'make -C "%s" libpy/libpy.so' % path
        out = os.system(command)
        if out:
            raise BuildFailed(
                "Command {!r} failed with code {}".format(command, out)
            )

        shutil.copyfile(
            'libpy/libpy.so',
            os.path.join(self.build_lib, 'libpy', 'libpy.so'),
        )

        p = pathlib.Path(self.build_lib) / 'libpy/_build-and-run'
        p.chmod(p.stat().st_mode | stat.S_IEXEC)


setup(
    name='libpy',
    description='Utilities for writing C++ extension modules.',
    long_description=open('README.rst').read(),
    url='https://github.com/quantopian/libpy',
    version=open('version').read().strip(),
    author='Quantopian Inc.',
    author_email='opensource@quantopian.com',
    packages=['libpy'],
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Topic :: Software Development',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: C++',
        'Operating System :: POSIX',
        'Intended Audience :: Developers',
    ],
    # we need the headers to be available to the C compiler as regular files;
    # we cannot be imported from a ziparchive.
    zip_safe=False,
    install_requires=['numpy'],
    cmdclass={'build_py': build_py},
    package_data={
        'libpy': [
            'include/libpy/*.h',
            'include/libpy/detail/*.h',
            '_build-and-run',
            '_detect-compiler.cc',
        ],
    },
)
