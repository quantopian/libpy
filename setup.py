import ast
import os
import sys

from setuptools import setup, find_packages


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
if 'build_ext' in sys.argv or 'egg_info' in sys.argv and not dont_build:
    path = os.path.dirname(os.path.abspath(__file__))
    command = 'make -C "%s"' % path
    out = os.system(command)
    if out:
        raise BuildFailed(
            "Command {!r} failed with code {}".format(command, out)
        )

setup(
    name='libpy',
    url='https://github.com/quantopian/libpy',
    author='Quantopian Inc.',
    author_email='opensource@quantopian.com',
    packages=find_packages(),
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Topic :: Software Development',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: C++',
        'Operating System :: POSIX',
        'Intended Audience :: Developers',
    ],
    # we need the headers to be available to the C compiler as regular files;
    # we cannot be imported from a ziparchive.
    zip_safe=False,
    include_package_data=True,
    install_requires=['numpy'],
    setup_requires=['setuptools_scm'],
    use_scm_version={
        'write_to': 'libpy/version.py',
        'write_to_template': '__version__ = "{version}"',
    },
)
