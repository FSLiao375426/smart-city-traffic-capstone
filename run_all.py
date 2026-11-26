"""Reproduce the whole capstone with one command (B3).

Runs every step in the order the parts depend on each other: tests, the Part 2
pipeline (it produces the data every part uses, D2), Part 1 analysis, Part 2
figures and CLI, then Part 3 models, monitoring and deployment demo. Stops at
the first step that fails and reports which one.

Run from the repository root:

    python run_all.py               # everything (about 5-10 minutes, mostly the LSTM)
    python run_all.py --skip-slow   # everything except the LSTM step
    python run_all.py --list        # show the steps without running them

Each step keeps its own log file (pipeline.log, part1.log, app.log, part3.log,
api.log). This script writes a short summary of the run to run_all.log.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent

# (description, module, arguments, slow)
STEPS = [
    ("Unit tests", "unittest", ["discover", "-s", "tests", "-t", ".", "-q"], False),
    ("Part 2: cleaning pipeline", "part2_python.pipeline", [], False),
    ("Part 2: feature engineering", "part2_python.feature_engineering", [], False),
    ("Part 2: figures", "part2_python.visualizations", [], False),
    ("Part 1: load SQLite", "part1_data_analytics.load_sqlite", [], False),
    ("Part 1: SQL analysis", "part1_data_analytics.run_sql_analysis", [], False),
    ("Part 1: statistics and probability", "part1_data_analytics.stats_probability", [], False),
    ("Part 2: CLI example", "part2_python.traffic_app", ["compare"], False),
    ("Part 3: supervised models", "part3_machine_learning.supervised", [], False),
    ("Part 3: clustering and association rules", "part3_machine_learning.unsupervised", [], False),
    ("Part 3: LSTM and SHAP", "part3_machine_learning.deep_learning", [], True),
    ("Part 3: monitoring", "part3_machine_learning.monitoring", [], False),
    ("Part 3: error distribution", "part3_machine_learning.fairness_analysis", [], False),
    ("Part 3: API demo", "part3_machine_learning.api_demo", [], False),
    ("Part 3: recommender example", "part3_machine_learning.recommender",
     ["--date", "2018-12-25", "--earliest", "8", "--latest", "18", "--window", "2"], False),
]


def run_step(number: int, total: int, description: str, module: str, args: list) -> float:
    """Run one step as `python -m module args`; raise RuntimeError if it fails."""
    command = [sys.executable, "-m", module, *args]
    logger.info("[%d/%d] %s: %s", number, total, description, " ".join(["python -m", module, *args]))
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-15:])
        logger.error("[%d/%d] %s failed with exit code %d after %.0fs. Last output:\n%s",
                     number, total, description, result.returncode, seconds, tail)
        raise RuntimeError(description)
    logger.info("[%d/%d] %s finished in %.0fs", number, total, description, seconds)
    return seconds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run every step of the capstone in order")
    parser.add_argument("--skip-slow", action="store_true", help="skip the LSTM step (the slowest)")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    args = parser.parse_args(argv)

    steps = [s for s in STEPS if not (args.skip_slow and s[3])]
    if args.list:
        for i, (description, module, extra, slow) in enumerate(steps, 1):
            print(f"{i:2d}. {description:42s} python -m {module} {' '.join(extra)}{'  (slow)' if slow else ''}")
        return 0

    setup_logging(ROOT / "run_all.log")
    logger.info("Running %d steps%s", len(steps), " (LSTM skipped)" if args.skip_slow else "")
    total_start = time.perf_counter()
    try:
        for i, (description, module, extra, _) in enumerate(steps, 1):
            run_step(i, len(steps), description, module, extra)
    except RuntimeError as exc:
        logger.error("Stopped at '%s'. Fix it and run again; see that step's own log for details.", exc)
        return 1
    logger.info("All %d steps finished in %.1f minutes", len(steps), (time.perf_counter() - total_start) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
