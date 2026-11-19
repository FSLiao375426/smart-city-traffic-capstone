"""One place to read the cleaned data, so every part reads it the same way."""

from pathlib import Path

import pandas as pd

from common import config


def load_clean(path: Path = config.CLEAN_CSV) -> pd.DataFrame:
    """Read traffic_clean.csv with the right types.

    pandas treats the text "None" as a missing value by default, which would
    turn every non-holiday hour into NaN. keep_default_na=False keeps "None"
    as the literal label written by the pipeline.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], parse_dates=["date_time"])
    df["holiday"] = df["holiday"].astype("string")
    return df
