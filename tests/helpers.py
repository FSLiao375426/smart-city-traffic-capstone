"""Small hand-made tables for the tests (no data files needed)."""

import pandas as pd


def raw_rows(rows):
    """Build a raw-format table from (date_time, weather_main, traffic_volume) tuples."""
    return pd.DataFrame({
        "holiday": "None",
        "temp": 280.0,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 40,
        "weather_main": [r[1] for r in rows],
        "weather_description": "test",
        "date_time": pd.to_datetime([r[0] for r in rows]),
        "traffic_volume": [r[2] for r in rows],
    })
