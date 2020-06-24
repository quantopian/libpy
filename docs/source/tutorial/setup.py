import ast
import glob
import os

from libpy.build import LibpyExtension
from setuptools import find_packages, setup

if ast.literal_eval(os.environ.get("LIBPY_TUTORIAL_DEBUG_BUILD", "0")):
    optlevel = 0
    debug_symbols = True
    max_errors = 5
else:
    optlevel = 3
    debug_symbols = False
    max_errors = None


def extension(*args, **kwargs):
    return LibpyExtension(
        *args,
        optlevel=optlevel,
        debug_symbols=debug_symbols,
        werror=True,
        max_errors=max_errors,
        include_dirs=["."] + kwargs.pop("include_dirs", []),
        depends=glob.glob("**/*.h", recursive=True),
        **kwargs
    )


install_requires = [
    'setuptools',
    'libpy',
    'matplotlib',
    'pillow',
]

setup(
    name="libpy_tutorial",
    version="0.1.0",
    description="Tutorial for libpy",
    author="Quantopian Inc.",
    author_email="opensource@quantopian.com",
    packages=find_packages(),
    package_data={
        "": ["*.png"],
    },
    include_package_data=True,
    install_requires=install_requires,
    license="Apache 2.0",
    url="https://github.com/quantopian/libpy",
    ext_modules=[
        extension(
            "libpy_tutorial.scalar_functions",
            ["libpy_tutorial/scalar_functions.cc"],
        ),
        extension(
            "libpy_tutorial.arrays",
            ["libpy_tutorial/arrays.cc"],
        ),
        extension(
            "libpy_tutorial.ndarrays",
            ["libpy_tutorial/ndarrays.cc"],
        ),
        extension(
            "libpy_tutorial.exceptions",
            ["libpy_tutorial/exceptions.cc"],
        ),
        extension(
            "libpy_tutorial.classes",
            ["libpy_tutorial/classes.cc"],
        ),
    ],
)
