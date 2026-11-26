"""Unit tests. Run from the repository root:

    python -m unittest discover -s tests -t . -v
"""

import logging

# Tests check log messages with assertLogs; this keeps other messages off the console
logging.getLogger().addHandler(logging.NullHandler())
