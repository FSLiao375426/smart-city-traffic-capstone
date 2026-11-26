"""One place to read the cleaned data, so every part reads it the same way."""

import logging
from pathlib import Path

import pandas as pd

from common import config

logger = logging.getLogger(__name__)


def load_clean(path: Path = config.CLEAN_CSV) -> pd.DataFrame:
    """Read traffic_clean.csv with the right types.

    pandas treats the text "None" as a missing value by default, which would
    turn every non-holiday hour into NaN. keep_default_na=False keeps "None"
    as the literal label written by the pipeline.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], parse_dates=["date_time"])
    for column in ["holiday", "weather_labels"]:
        if column in df.columns:
            df[column] = df[column].astype("string")
    logger.debug("Read %s: %d rows, %d columns", Path(path).name, *df.shape)
    return df


def load_features(path: Path = config.FEATURES_CSV) -> pd.DataFrame:
    """Read traffic_features.csv with the same "None" handling as load_clean()."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], parse_dates=["date_time"])
    for column in ["holiday", "weather_main", "weather_labels", "weather_description", "day_name",
                   "day_type", "time_of_day", "traffic_category", "congestion_category"]:
        df[column] = df[column].astype("string")
    logger.debug("Read %s: %d rows, %d columns", Path(path).name, *df.shape)
    return df
