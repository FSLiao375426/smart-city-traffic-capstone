"""Logging configuration, called ONLY from entry-point scripts.

Library modules never configure logging. They only do:

    import logging
    logger = logging.getLogger(__name__)

An entry point (pipeline.py, the CLI app, a training script) calls
setup_logging() once at start-up. Messages then go to two places:

* the console, at the chosen level (INFO by default, DEBUG with --debug)
* a log file, so graders can read the trail without re-running anything

Level meanings used across the project:
    DEBUG   fine-grained internal values (e.g. quartile thresholds)
    INFO    normal milestones (data loaded, step finished, file saved)
    WARNING unexpected but recoverable (rows dropped, values imputed)
    ERROR   the run cannot continue as planned
"""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: Path, debug: bool = False, append: bool = False,
                  console_level: int = None) -> None:
    """Attach a console handler and a file handler to the root logger.

    Args:
        log_file: where the log is written (overwritten on every run).
        debug: if True, DEBUG messages are shown; otherwise INFO and above.
        append: if True, add to an existing log (for multi-script workflows);
            otherwise start a fresh log file.
        console_level: optional separate level for the console, e.g. WARNING for
            a command-line app whose screen should show only answers and errors.
    """
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    if console_level is not None:
        console.setLevel(console_level)

    file_handler = logging.FileHandler(log_file, mode="a" if append else "w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()          # avoid duplicate lines if called twice
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
