from . import _test_automodule as mod


def test_modname():
    assert mod.__name__ == 'tests._test_automodule'


def test_function():
    assert mod.is_42(42)
    assert not mod.is_42(~42)


def test_type():
    assert isinstance(mod.int_float_pair, type)
    a = mod.int_float_pair(1, 2.5)
    assert a.first() == 1
    assert a.second() == 2.5
    b = mod.int_float_pair(1, 2.5)
    assert a == b
    c = mod.int_float_pair(1, 3.5)
    assert a != c
