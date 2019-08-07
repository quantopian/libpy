def pytest_addoption(parser):
    """Add new pytest options.
    """
    parser.addoption('--fuzz-parser-root-seed', type=int, default=None)
    parser.addoption('--fuzz-parser-num-iterations', type=int, default=10)
    parser.addoption('--fuzz-parser-num-rows', type=int, default=100)
    # This defaults to 1 because we need at least 10000 lines in the csv to use
    # threads anyway.
    parser.addoption('--fuzz-parser-num-threads', type=int, default=1)
