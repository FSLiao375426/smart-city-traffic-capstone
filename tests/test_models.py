"""Part 3: chronological split, model selection rule, recommender input checks, safe rates."""

import math
import unittest

import pandas as pd

from part3_machine_learning import ml_common as mc
from part3_machine_learning.fairness_analysis import safe_rate
from part3_machine_learning.recommender import RecommendationError, history_support, recommend
from part3_machine_learning.supervised import choose_model


class SplitTest(unittest.TestCase):
    """D19: training to 2016, validation 2017, test 2018, no overlap."""

    def test_three_periods_in_order_without_overlap(self):
        df = pd.DataFrame({"date_time": pd.date_range("2015-12-31", "2018-01-02", freq="6h")})
        train, valid, test = mc.three_way_split(df)
        self.assertEqual(len(train) + len(valid) + len(test), len(df))
        self.assertLess(train["date_time"].max(), valid["date_time"].min())
        self.assertLess(valid["date_time"].max(), test["date_time"].min())
        self.assertEqual(valid["date_time"].dt.year.unique().tolist(), [2017])


class SelectionRuleTest(unittest.TestCase):
    def test_best_validation_score_wins(self):
        scores = {"dummy": {"MAE": 1700}, "linear": {"MAE": 800},
                  "random_forest": {"MAE": 300}, "hist_gradient_boosting": {"MAE": 240}}
        self.assertEqual(choose_model("regression", scores)[0], "hist_gradient_boosting")

    def test_a_near_tie_goes_to_the_simpler_model(self):
        scores = {"dummy": {"MAE": 1700}, "linear": {"MAE": 800},
                  "random_forest": {"MAE": 236}, "hist_gradient_boosting": {"MAE": 240}}   # 1.7% apart
        choice, reason = choose_model("regression", scores)
        self.assertEqual(choice, "hist_gradient_boosting")
        self.assertIn("simpler", reason)

    def test_dummy_is_never_chosen(self):
        scores = {"dummy": {"F1": 0.99}, "linear": {"F1": 0.5},
                  "random_forest": {"F1": 0.6}, "hist_gradient_boosting": {"F1": 0.6}}
        self.assertNotEqual(choose_model("classification", scores)[0], "dummy")


class RecommenderInputTest(unittest.TestCase):
    """Bad input is rejected before any model or data file is loaded."""

    def test_malformed_date(self):
        with self.assertRaises(RecommendationError):
            recommend("14/11/2018")

    def test_window_longer_than_range(self):
        with self.assertRaises(RecommendationError):
            recommend("2018-11-14", earliest=8, latest=9, window=3)

    def test_unknown_weather(self):
        with self.assertRaises(RecommendationError):
            recommend("2018-11-14", weather="Hail")

    def test_history_support_counts_matching_hours(self):
        idx = pd.MultiIndex.from_tuples([(0, 7), (0, 8), (0, 9), (1, 8)], names=["is_weekend", "hour"])
        support = {"Snow": pd.Series([10, 20, 30, 5], index=idx)}
        self.assertEqual(history_support(support, "Snow", 0, 7, 8), 30)
        self.assertEqual(history_support(support, "Snow", 1, 6, 10), 5)


class SafeRateTest(unittest.TestCase):
    def test_empty_denominator_is_missing_not_zero(self):
        self.assertTrue(math.isnan(safe_rate(0, 0)))
        self.assertEqual(safe_rate(1, 4), 0.25)


if __name__ == "__main__":
    unittest.main()
