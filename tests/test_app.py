"""Part 2 CLI: user input is checked and errors are reported clearly."""

import unittest

import pandas as pd

from part2_python import traffic_app as app


class InputParsingTest(unittest.TestCase):
    def test_accepted_date_time_formats(self):
        expected = pd.Timestamp("2017-06-01 08:00")
        for text in ["2017-06-01 08:00", "2017-06-01 08:00:00", "2017-06-01 08"]:
            self.assertEqual(app.parse_datetime(text), expected)

    def test_malformed_date_time_raises_a_user_error(self):
        for text in ["01/06/2017 8am", "2017-13-01 08:00", ""]:
            with self.assertRaises(app.UserInputError):
                app.parse_datetime(text)

    def test_hour_outside_0_to_23_is_rejected(self):
        self.assertEqual(app.check_hour(23, "latest"), 23)
        with self.assertRaises(app.UserInputError):
            app.check_hour(24, "latest")


if __name__ == "__main__":
    unittest.main()
