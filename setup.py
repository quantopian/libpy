import os
import sys

from setuptools import setup, find_packages


class BuildFailed(Exception):
    pass


if 'build_ext' in sys.argv or 'egg_info' in sys.argv:
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
    version='0.1.0',
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
)
