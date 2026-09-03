import pytest
import pandas as pd
import numpy as np
from model.features import FeatureBuilder


def _synthetic_player_game_stats(n_weeks=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, n_weeks + 1):
        rows.append({
            "player_id": "TEST-RB-1", "player_name": "Test Runner", "position": "RB",
            "team": "KC", "opponent": "SF" if week % 2 == 0 else "DEN",
            "season": 2024, "week": week, "home_game": week % 2 == 0,
            "rushing_yards": int(rng.normal(70, 20)), "rushing_attempts": int(rng.normal(15, 3)),
            "receiving_yards": int(rng.normal(15, 5)), "receptions": 2, "targets": 3,
            "target_share": 0.1, "air_yards_share": 0.05, "wopr": 0.15,
            "passing_yards": 0, "passing_attempts": 0,
            "rushing_tds": 0, "receiving_tds": 0, "passing_tds": 0,
        })
        # A second player on the opposing side, so opponent-defense aggregation has data
        rows.append({
            "player_id": "TEST-RB-2", "player_name": "Test Opponent", "position": "RB",
            "team": "SF" if week % 2 == 0 else "DEN", "opponent": "KC",
            "season": 2024, "week": week, "home_game": week % 2 != 0,
            "rushing_yards": int(rng.normal(60, 15)), "rushing_attempts": int(rng.normal(12, 3)),
            "receiving_yards": int(rng.normal(10, 5)), "receptions": 1, "targets": 2,
            "target_share": 0.08, "air_yards_share": 0.04, "wopr": 0.1,
            "passing_yards": 0, "passing_attempts": 0,
            "rushing_tds": 0, "receiving_tds": 0, "passing_tds": 0,
        })
    return pd.DataFrame(rows)


class TestFeatureBuilder:

    def setup_method(self):
        self.df = _synthetic_player_game_stats()
        self.fb = FeatureBuilder("rushing_yards")

    def test_build_produces_expected_columns(self):
        features_df, feature_cols = self.fb.build(self.df)
        assert "roll3" in feature_cols
        assert "opp_def_avg_allowed" in feature_cols
        for col in feature_cols:
            assert col in features_df.columns

    def test_build_drops_week1_rows(self):
        # Week 1 has no prior games, so games_played == 0 and it should be excluded.
        features_df, _ = self.fb.build(self.df)
        test_player_rows = features_df[features_df["player_id"] == "TEST-RB-1"]
        assert 1 not in test_player_rows["week"].values

    def test_rolling_features_have_no_leakage(self):
        # roll3 for week 3 must equal the mean of weeks 1-2's actual values,
        # never including week 3's own value.
        features_df, _ = self.fb.build(self.df)
        player_df = self.df[self.df["player_id"] == "TEST-RB-1"].sort_values("week")
        week3_roll3 = features_df[
            (features_df["player_id"] == "TEST-RB-1") & (features_df["week"] == 3)
        ]["roll3"].iloc[0]
        expected = player_df[player_df["week"].isin([1, 2])]["rushing_yards"].mean()
        assert abs(week3_roll3 - expected) < 0.01

    def test_build_for_prediction_returns_single_row(self):
        feature_row, feature_cols = self.fb.build_for_prediction(
            self.df, schedules_df=None,
            player_id="TEST-RB-1", season=2024, week=11, team="KC", opponent="DEN", home_game=True,
        )
        assert len(feature_row) == len(feature_cols)
        assert feature_row["games_played"] == 10  # all 10 synthetic weeks precede week 11

    def test_build_for_prediction_unknown_player_raises(self):
        with pytest.raises(ValueError):
            self.fb.build_for_prediction(
                self.df, schedules_df=None,
                player_id="NOT-A-REAL-PLAYER", season=2024, week=11, team="KC", opponent="DEN", home_game=True,
            )


if __name__ == "__main__":
    pytest.main([__file__])
