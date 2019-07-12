import sys

import _runner


exit(int(_runner.run_tests(tuple(arg.encode() for arg in sys.argv))))
