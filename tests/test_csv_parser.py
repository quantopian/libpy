import io
import random
import string
import sys
from operator import and_
from textwrap import dedent

import numpy as np

from . import cxx

if sys.version_info.major > 2:
    from functools import reduce


def isnat(x):
    return x == np.datetime64('nat').astype(x.dtype)


def assert_table_equal(actual, expected):
    assert sorted(actual) == sorted(expected)

    for k, actual_col in actual.items():
        expected_col = expected[k]
        np.testing.assert_array_equal(actual_col, expected_col)


def count(table):
    return len(list(table.values()).pop())


def _generate_datetime64D(rand, count):
    return rand.randint(
        -365 * 30,
        365 * 30,
        size=count,
    ).view('M8[D]')


def _generate_datetime64s(rand, count):
    return rand.randint(
        -365 * 30 * 36000,
        365 * 30 * 36000,
        size=count,
    ).view('M8[s]')


def _generate_f(rand, count, dtype):
    return rand.uniform(
        low=-10e10,
        high=10e10,
        size=count,
    ).astype(dtype)


def _generate_i(rand, count, dtype):
    limits = np.iinfo(dtype)
    return rand.uniform(
        low=limits.min,
        high=limits.max,
        size=count,
    ).astype(dtype)


def _generate_bool(rand, count):
    return rand.randint(
        low=0,
        high=2,  # exclusive
        size=count,
        dtype='i1',
    )


def _generate_S(rand, count, size):
    char_choices = np.array(
        sorted(set(string.printable) - set(string.whitespace)),
        dtype='S1',
    )
    return rand.choice(
        char_choices,
        size=count * size,
    ).view(('S', size))


def _generate_O(rand, count):
    """Generate an array of random uppercase strings for a numpy bytes dtype.
    """
    base = np.array(
        [c.encode() for c in string.ascii_letters + string.digits],
        dtype='S1',
    )

    return np.array(
        [
            b''.join(rand.choice(base, rand.randint(3, 12)))
            for _ in range(count)
        ],
        dtype='O',
    )


def _generate_column(name, dtype, rand, count):
    if dtype == np.dtype('M8[D]'):
        return _generate_datetime64D(rand, count).astype('M8[ns]')
    if dtype == np.dtype('M8[s]'):
        return _generate_datetime64s(rand, count).astype('M8[ns]')
    if dtype.kind == 'f':
        return _generate_f(rand, count, dtype)
    if dtype.kind in 'ui':
        return _generate_i(rand, count, dtype)
    if dtype.kind == 'b':
        return _generate_bool(rand, count)
    if dtype.kind == 'S':
        return _generate_S(rand, count, dtype.itemsize)
    if dtype.kind == 'O':
        return _generate_O(rand, count)

    raise TypeError('unknown dtype: %s' % dtype)


def generate_table(rand, schema, rows):
    """Generate a random table.

    Parameters
    ----------
    rand : np.random.RandomState
        The source of randomness.
    schema : dict[str, np.dtype]
        The names and dtypes of the columns.
    rows : int
        The number of rows to generate.

    Returns
    -------
    table : dict[str, np.ndarray]
        The table with the same dtype as ``schema``.
    mask : dict[str, np.ndarray[?]]
        A table of all booleans with the same columns as ``table``. This mask
        will be ~5% False.
    """
    table = {
        name: _generate_column(name, dtype, rand, rows)
        for name, dtype in schema.items()
    }

    # 5% missing values
    dense_mask = rand.uniform(size=(len(schema), rows)) > 0.05
    mask = {
        name: dense_mask[ix]
        for ix, name in enumerate(schema)
    }

    return table, mask


def _format_str(cs):
    escaped = cs.decode('ascii').replace('\\', '\\\\').replace('"', r'\"')
    return '"%s"' % escaped


def _format_date(dt):
    if isnat(dt):
        return ''
    return dt.astype('M8[D]').item().strftime('%Y-%m-%d')


def _format_datetime(dt):
    if isnat(dt):
        return ''
    return str(dt)[:-len('.000000000')]


def _format_value(dtype, v, valid):
    if not valid:
        return ''

    if isinstance(v, bytes):
        return _format_str(v)
    if dtype == np.dtype('M8[D]'):
        return _format_date(v)
    if dtype == np.dtype('M8[s]'):
        return _format_datetime(v)
    return repr(v)


def to_csv(table, mask=None, schema=None):
    """
    Convert a table and mask into a CSV.

    Parameters
    ----------
    table : Table
        The table to write.
    mask : Table, optional
        A table of all booleans which indicate which cells should be
        written. By default, all values in ``table`` will be written.
    schema : dict[str, np.dtype], optional
        The table schema. This can be used to explicitly override the dtype of
        ``table`` (e.g., to convert datetimes to dates). Default is to use
        ``table.dtypes``.

    Returns
    -------
    file : file-like
        The written CSV file at the beginning of the stream.
    """
    out = io.BytesIO()
    write_csv_into(out, table, schema=schema, mask=mask)
    out.seek(0)
    return out


def write_csv_into(out, table, mask=None, schema=None):
    """
    Write a table into ``out`` as a csv.

    Parameters
    ----------
    out : file-like
        File-like object into which to write serialized table.
    table : Table
        The table to write.
    mask : Table, optional
        A table of all booleans which indicate which cells should be
        written. If not passed, all values in ``table`` will be written.
    schema : dict[str, np.dtype], optional
        The table schema. If not passed, the table's dtypes are used unchanged.
        This can be used to differentiate dates and datetimes.
    """
    if mask is None:
        all_true = np.full((count(table),), True)
        mask = {k: all_true for k in table.keys()}

    if schema is None:
        schema = {k: v.dtype for k, v in table.items()}

    column_order = list(table)
    header = '|'.join(column_order).encode('latin-1')
    out.write(header)
    out.write(b'\r\n')

    for n in range(count(table)):
        row = '|'.join(
            _format_value(schema[column], table[column][n], mask[column][n])
            for column in column_order
        ).encode('latin-1')
        out.write(row)
        out.write(b'\r\n')


def table_and_csv(rand, schema, rows):
    """Generate a table and the matching CSV.

    Parameters
    ----------
    rand : np.random.RandomState
        The source of randomness.
    schema : dict[str, np.dtype]
        The names and dtypes of the columns.
    rows : int
        The number of rows to generate.

    Returns
    -------
    table : Table
        The table with the same dtype as ``schema``.
    mask : Table
        A table of all booleans with the same columns as ``table``. This mask
        will be ~5% False.
    csv_file : file-like
        The written CSV file at the beginning of the stream.
    """
    table, mask = generate_table(rand, schema, rows)
    csv_file = to_csv(table=table, mask=mask, schema=schema)
    return table, mask, csv_file


def parse_csv(data,
              schema,
              required_columns,
              field_separator,
              line_separator,
              num_threads):
    """
    Parse a CSV from a string.

    Parameters
    ----------
    data : bytes
        CSV data to be parsed.
    schema : dict[bytes, np.dtype]
        Map from column name to expected column type.
    required_columns : list[bytes]
        List of columns that must parse to a nonempty value. Any rows with NULL
        values in one or more of these columns will be discarded.
    field_separator : bytes
        Characters used to separate entries within a line.
    line_separator : bytes
        Characters used to separater lines.
    num_threads : int
        Number of threads to use during parsing.

    Returns
    -------
    parsed : Table
        Table containing data parsed from ``data``.
    """
    required_columns = set(required_columns)
    missing = required_columns - set(schema.keys())
    if missing:
        raise ValueError(
            "Required column(s) %s not in %s." % (
                sorted(required_columns),
                sorted(schema.keys()),
            )
        )

    parsed = cxx.parse_csv(
        data,
        schema,
        field_separator,
        line_separator,
        num_threads,
    )

    columns = {}
    masks = []
    for name, (column, mask) in parsed.items():
        columns[name] = column
        if name in required_columns:
            masks.append(mask)

    table = columns
    if not count(table):
        return table

    # Filter down to only rows that are well-formed.
    if masks:
        mask = reduce(and_, masks)
        table.filter_inplace(mask)

    return table


MISSING_VALUES = {
    np.dtype('datetime64[D]'): np.datetime64('NaT', 'D'),
    np.dtype('datetime64[s]'): np.datetime64('NaT', 's'),
    np.dtype('datetime64[ns]'): np.datetime64('NaT', 'ns'),
    np.dtype('float64'): np.nan,
    np.dtype('int8'): 0,
    np.dtype('S1'): b'\0',
    np.dtype('S6'): b'\0' * 6,
    np.dtype('O'): None,
}
# TODO: Is there a better way to do this?
#
# We don't ever parse anything with datetime64[ns] because factset never gives
# us that much precision, but we need an entry in MISSING_VALUES table_and_csv
# always produces datetime64[ns] for its output columns.
DTYPES = sorted(set(MISSING_VALUES.keys()) - {np.dtype('datetime64[ns]')})
COLNAMES = list(string.ascii_uppercase[:20])

DTYPE_TO_COLUMN_SPEC = {
    np.dtype('datetime64[D]'): cxx.DateTime(b'y-m-d'),
    np.dtype('datetime64[s]'): cxx.DateTime(b'y-m-d h:m:s'),
    np.dtype('datetime64[ns]'): cxx.DateTime(b'y-m-d h:m:s.s'),
    np.dtype('float64'): cxx.Float64(b'precise'),
    np.dtype('int8'): cxx.Int8(),
    np.dtype('S1'): cxx.String(1),
    np.dtype('S6'): cxx.String(6),
    np.dtype('O'): cxx.String(-1),
}


def check_parse(csv_data, schema, expected_result, num_threads):
    result = parse_csv(
        csv_data,
        schema={
            k.encode(): DTYPE_TO_COLUMN_SPEC[v] for k, v in schema.items()
        },
        required_columns=[],
        field_separator=b'|',
        line_separator=b'\r\n',
        num_threads=1,
    )
    assert_table_equal(result, expected_result)


def test_fuzz_parser(request):
    root_seed = request.config.getoption('--fuzz-parser-root-seed')
    if root_seed is None:
        # NOTE: Normally it's desirable for random data in a test to be
        # generated deterministically. In this case, however, our goal is to
        # fuzz the CSV parser, so we want to run with different inputs every
        # time. In the event of a failure, the error message should include
        # enough information to reproduce the failure deterministically.
        root_seed = random.SystemRandom().randint(0, 10000000)
    rand = np.random.RandomState(root_seed)

    # These are defined in factset_utils/conftest.py. Override these values
    # manually on the command line to fuzz more aggressively.
    num_iterations = request.config.getoption('--fuzz-parser-num-iterations')
    nrows = request.config.getoption('--fuzz-parser-num-rows')
    num_threads = request.config.getoption('--fuzz-parser-num-threads')

    for i in range(num_iterations):
        ncols = rand.randint(1, len(COLNAMES) + 1)
        colnames = rand.choice(COLNAMES, ncols, replace=False)
        dtypes = rand.choice(DTYPES, ncols, replace=True)
        schema = dict(zip(colnames, dtypes))

        table, mask, csv_file = table_and_csv(rand, schema, nrows)
        table = {k.encode(): v for k, v in table.items()}
        mask = {k.encode(): v for k, v in mask.items()}

        # Expect to parse missing values from masked locations.
        expected = table.copy()
        for colname, arr in expected.items():
            expected[colname][~mask[colname]] = MISSING_VALUES[arr.dtype]

        csv_data = csv_file.read()

        try:
            check_parse(csv_data, schema, expected, num_threads)
        except AssertionError:
            raise AssertionError(
                "Parse result did not match expectations for "
                "root_seed=%s, iteration=%s." % (root_seed, i)
            )


def test_empty_string_vs_missing_string_vlen():
    data = dedent(
        """\
        "a","b"
        ,,
        "",
        ,""
        """,
    ).encode()

    result = cxx.parse_csv(
        data,
        column_specs={b'a': cxx.String(-1), b'b': cxx.String(-1)},
        delimiter=b',',
        line_ending=b'\n',
        num_threads=1,
    )
    expected = {
        b'a': (
            np.array([None, b'', None]),
            np.array([False, True, False]),
        ),
        b'b': (
            np.array([None, None, b'']),
            np.array([False, False, True]),
        ),
    }
    assert result.keys() == expected.keys()
    for k in b'a', b'b':
        expected_data, expected_mask = expected[k]
        actual_data, actual_mask = result[k]

        np.testing.assert_array_equal(expected_data, actual_data)
        np.testing.assert_array_equal(expected_mask, actual_mask)


def test_empty_string_vs_missing_string_fixed():
    data = dedent(
        """\
        "a","b"
        ,,
        "",
        ,""
        """,
    ).encode()

    result = cxx.parse_csv(
        data,
        column_specs={b'a': cxx.String(3), b'b': cxx.String(3)},
        delimiter=b',',
        line_ending=b'\n',
        num_threads=1,
    )
    expected = {
        b'a': (
            np.array([b'', b'', b''], dtype='S3'),
            np.array([False, True, False]),
        ),
        b'b': (
            np.array([b'', b'', b''], dtype='S3'),
            np.array([False, False, True]),
        ),
    }
    assert result.keys() == expected.keys()
    for k in b'a', b'b':
        expected_data, expected_mask = expected[k]
        actual_data, actual_mask = result[k]

        np.testing.assert_array_equal(expected_data, actual_data)
        np.testing.assert_array_equal(expected_mask, actual_mask)
