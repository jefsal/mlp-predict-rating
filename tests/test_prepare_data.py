"""Small checks for the forecasting boundary that matters to this project."""

import importlib.util
from pathlib import Path
import unittest

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ForecastTests(unittest.TestCase):
    def sample(self):
        rows = []
        for index in range(7):
            row = {column: float(index + 1) for column in prepare.HISTORY_STATS}
            row.update(
                player_id=1, match_id=f"match-{index}",
                kickoff_time=pd.Timestamp("2025-08-01", tz="UTC") + pd.Timedelta(days=index),
                first_name="Test", second_name="Player", position="Forward",
                minutes_played=90,
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def test_target_and_future_outcomes_do_not_change_earlier_features(self):
        data = self.sample()
        before = prepare.build_forecasts(data)
        data.loc[5:, prepare.HISTORY_STATS] = 8.0
        after = prepare.build_forecasts(data)
        columns = [*prepare.FEATURES, "past_5_rating_count"]
        pd.testing.assert_series_equal(before.iloc[0][columns], after.iloc[0][columns])
        self.assertEqual(before.iloc[0]["past_5_goals_mean"], 3.0)
        self.assertNotIn("minutes_played", before.columns)

    def test_unrated_appearance_still_contributes_stats_to_history(self):
        data = self.sample()
        data.loc[4, "rating"] = float("nan")
        result = prepare.build_forecasts(data)
        self.assertEqual(result.iloc[0]["past_5_rating_count"], 4)
        self.assertEqual(result.iloc[0]["past_5_goals_mean"], 3.0)

    def test_zero_minutes_and_missing_targets_are_excluded(self):
        data = self.sample()
        data.loc[5, "minutes_played"] = 0
        result = prepare.build_forecasts(data)
        self.assertEqual(result["match_id"].tolist(), ["match-6"])
        data.loc[6, "rating"] = float("nan")
        self.assertTrue(prepare.build_forecasts(data).empty)


if __name__ == "__main__":
    unittest.main()
