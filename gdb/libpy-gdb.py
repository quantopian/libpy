import re

import gdb
import numpy as np


def pretty_printer(cls):
    gdb.pretty_printers.append(cls.maybe_construct)
    return cls


@pretty_printer
class Datetime64:
    _pattern = re.compile(
        r'^py::datetime64<std::chrono::duration<long,'
        r' std::ratio<(\d+), (\d+)> > >$'
    )

    _units = {
        (1, 1000000000): 'ns',
        (1, 1000000): 'ms',
        (1, 1000): 'us',
        (1, 1): 's',
        (60, 1): 'm',
        (60 * 60, 1): 'h',
        (60 * 60 * 24, 1): 'D',
    }

    def __init__(self, val, count, unit):
        self.val = val
        self.count = count
        self.unit = unit

    @classmethod
    def maybe_construct(cls, val):
        underlying = gdb.types.get_basic_type(val.type)

        match = cls._pattern.match(str(underlying))
        if match is None:
            return None

        num = int(match[1])
        den = int(match[2])
        return cls(val, int(val['m_value']['__r']), cls._units[num, den])

    def children(self):
        yield 'm_value', str(np.datetime64(self.count, self.unit))

    def to_string(self):
        return f'py::datetime64<py::chrono::{self.unit}>'
