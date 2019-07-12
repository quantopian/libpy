import sys

import _runner

out = int(_runner.run_tests(tuple(arg.encode() for arg in sys.argv)))
exit(out)
