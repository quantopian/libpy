import sys

import _runner


exit(_runner.run_tests(tuple(arg.encode() for arg in sys.argv)))
