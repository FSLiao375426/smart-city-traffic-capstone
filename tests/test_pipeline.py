"""Part 2 pipeline: schema check, hourly collapse, imputation, flags and holidays."""

import unittest

import pandas as pd

from common import config
from part2_python import pipeline
from tests.helpers import raw_rows


class SchemaTest(unittest.TestCase):
    def test_missing_column_stops_the_pipeline(self):
        df = raw_rows([("2016-01-01 08:00", "Clear", 3000)]).drop(columns="traffic_volume")
        with self.assertRaises(pipeline.PipelineError):
            pipeline.validate_schema(df)

    def test_complete_schema_passes(self):
        df = raw_rows([("2016-01-01 08:00", "Clear", 3000)])
        with self.assertLogs(pipeline.logger, level="INFO"):
            pipeline.validate_schema(df)


class ParseDatetimeTest(unittest.TestCase):
    def test_unparseable_rows_are_dropped_with_a_warning(self):
        df = raw_rows([("2016-01-01 08:00", "Clear", 3000)])
        df["date_time"] = ["2016-01-01 08:00:00"]
        df = pd.concat([df, df.assign(date_time="not a date")], ignore_index=True)
        with self.assertLogs(pipeline.logger, level="WARNING"):
            out = pipeline.parse_datetimes(df)
        self.assertEqual(len(out), 1)


class HourlyCollapseTest(unittest.TestCase):
    """D1 and D21: one row per hour, traffic not summed, every weather label kept."""

    def setUp(self):
        self.df = raw_rows([
            ("2016-03-01 08:00", "Mist", 5000),
            ("2016-03-01 08:00", "Rain", 5000),
            ("2016-03-01 08:00", "Mist", 5000),
            ("2016-03-01 09:00", "Clear", 4000),
        ])
        self.df["weather_description"] = ["mist", "light rain", "haze-like mist", "sky is clear"]

    def test_one_row_per_hour_and_traffic_not_summed(self):
        out = pipeline.collapse_to_hourly(self.df)
        self.assertTrue(out["date_time"].is_unique)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["traffic_volume"].sum(), 9000)      # 5000 + 4000, not 3 x 5000 + 4000

    def test_most_severe_label_kept_and_all_labels_listed(self):
        out = pipeline.collapse_to_hourly(self.df).set_index("date_time")
        eight = out.loc[pd.Timestamp("2016-03-01 08:00")]
        self.assertEqual(eight["weather_main"], "Rain")         # rain ranks above mist
        self.assertEqual(eight["weather_labels"], "Rain|Mist")  # distinct labels, most severe first
        self.assertEqual(eight["weather_label_count"], 3)


class ImputationTest(unittest.TestCase):
    def test_monthly_median_uses_the_same_month_only(self):
        df = raw_rows([("2016-01-05 08:00", "Clear", 1), ("2016-01-06 08:00", "Clear", 1),
                       ("2016-01-07 08:00", "Clear", 1), ("2016-07-05 08:00", "Clear", 1)])
        df["temp"] = [260.0, 0.0, 262.0, 300.0]                   # 0 K is impossible
        invalid = df["temp"] < config.TEMP_MIN_K
        with self.assertLogs(pipeline.logger, level="WARNING"):
            out = pipeline.impute_with_monthly_median(df, "temp", invalid, "test")
        self.assertEqual(out.loc[1, "temp"], 261.0)               # January median, not including July

    def test_nothing_changes_when_all_values_are_valid(self):
        df = raw_rows([("2016-01-05 08:00", "Clear", 1)])
        out = pipeline.impute_with_monthly_median(df, "temp", df["temp"] < 0, "test")
        pd.testing.assert_frame_equal(out, df)


class AnomalyAndHolidayTest(unittest.TestCase):
    def test_low_daytime_traffic_is_flagged_but_not_changed(self):
        df = raw_rows([("2016-07-23 12:00", "Clear", 5), ("2016-07-23 03:00", "Clear", 5),
                       ("2016-07-23 13:00", "Clear", 4000)])
        with self.assertLogs(pipeline.logger, level="WARNING"):
            out = pipeline.flag_traffic_anomalies(df)
        self.assertEqual(out["traffic_anomaly"].tolist(), [1, 0, 0])   # night-time low traffic is normal
        self.assertEqual(out["traffic_volume"].tolist(), [5, 5, 4000])  # the target is never imputed

    def test_holiday_label_extends_to_the_whole_day(self):
        df = raw_rows([("2016-12-25 00:00", "Clear", 800), ("2016-12-25 10:00", "Clear", 2000),
                       ("2016-12-26 10:00", "Clear", 3000)])
        df["holiday"] = pd.Series(["Christmas Day", "None", "None"], dtype="string")
        with self.assertLogs(pipeline.logger, level="WARNING"):
            out = pipeline.extend_holidays(df)
        self.assertEqual(out["is_holiday"].tolist(), [1, 1, 0])
        self.assertEqual(out.loc[1, "holiday"], "Christmas Day")


if __name__ == "__main__":
    unittest.main()
