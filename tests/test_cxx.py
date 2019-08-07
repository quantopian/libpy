import os
import warnings

from . import cxx


def test_cxx(capfd):
    # capfd.disabled() will let gtest print its own output
    with warnings.catch_warnings(), capfd.disabled():
        warnings.simplefilter('ignore')
        assert not cxx.run_tests(tuple(
            arg.encode() for arg in os.environ.get('GTEST_ARGS', '').split()
        ))
