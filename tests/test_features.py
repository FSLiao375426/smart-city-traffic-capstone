"""Part 2 feature engineering: cyclical time, multi-label weather, training-only thresholds."""

import unittest

import numpy as np
import pandas as pd

from common import config
from part2_python import feature_engineering as fe


class CyclicalTest(unittest.TestCase):
    def test_hour_23_and_hour_0_are_neighbours(self):
        sin, cos = fe.cyclical(pd.Series(range(24)), 24)
        self.assertTrue(((sin >= -1) & (sin <= 1) & (cos >= -1) & (cos <= 1)).all())
        gap_23_0 = np.hypot(sin[23] - sin[0], cos[23] - cos[0])
        gap_0_12 = np.hypot(sin[12] - sin[0], cos[12] - cos[0])
        self.assertLess(gap_23_0, 0.3)          # one hour apart on the clock
        self.assertAlmostEqual(gap_0_12, 2.0)   # opposite sides of the circle


class WeatherFeaturesTest(unittest.TestCase):
    """D21: an hour with rain and mist is both precipitation and low visibility."""

    def setUp(self):
        self.df = pd.DataFrame({
            "temp": [280.0, 280.0, 280.0],
            "weather_main": ["Rain", "Clear", "Snow"],
            "weather_labels": ["Rain|Mist", "Clear", "Snow"],
            "rain_1h": [1.0, 0.0, 0.0],
            "snow_1h": [0.0, 0.0, 0.0],
        })

    def test_every_label_counts(self):
        out = fe.add_weather_features(self.df)
        self.assertEqual(out["is_low_visibility"].tolist(), [1, 0, 0])
        self.assertEqual(out["is_precipitation"].tolist(), [1, 0, 1])
        self.assertEqual(out["is_severe_weather"].tolist(), [0, 0, 1])
        self.assertEqual(out.loc[0, ["wx_Rain", "wx_Mist", "wx_Clear"]].tolist(), [1, 1, 0])

    def test_same_weather_columns_in_every_run(self):
        out = fe.add_weather_features(self.df)
        expected = {f"wx_{w}" for w in config.WEATHER_SEVERITY}
        self.assertEqual({c for c in out.columns if c.startswith("wx_")}, expected)


class TargetsTest(unittest.TestCase):
    """D20: quartile thresholds come from the training period only."""

    def setUp(self):
        train = pd.date_range("2016-01-01", periods=100, freq="h")
        later = pd.date_range("2017-06-01", periods=20, freq="h")
        self.df = pd.DataFrame({
            "date_time": train.append(later),
            "traffic_volume": list(range(100)) + [10_000] * 20,   # later years are much busier
            "is_severe_weather": 0,
            "is_low_visibility": 0,
        })

    def test_thresholds_ignore_validation_and_test_years(self):
        params = {}
        fe.add_targets(self.df, params)
        q = params["congestion_quartiles"]
        self.assertEqual((q["q1"], q["q2"], q["q3"]), (24.75, 49.5, 74.25))   # quartiles of 0..99 only

    def test_high_risk_needs_both_heavy_traffic_and_risky_weather(self):
        df = self.df.copy()
        df.loc[[10, 90], "is_low_visibility"] = 1       # light traffic + fog, heavy traffic + fog
        df.loc[95, "is_severe_weather"] = 0             # heavy traffic, clear
        out = fe.add_targets(df, {})
        self.assertEqual(out.loc[[10, 90, 95], "high_risk"].tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
