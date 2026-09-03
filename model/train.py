import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from model.features import FeatureBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("model/artifacts")


class PropModelTrainer:

    def __init__(self, stat_column):
        self.stat_column = stat_column
        self.feature_builder = FeatureBuilder(stat_column)
        self.model = None
        self.residual_std = None
        self.feature_cols = None

    # Time-based split: train on earlier seasons, test on the most recent one.
    # A random split would leak future information into training (e.g. training on
    # a player's week 10 while testing his week 3 from the same season) and make
    # backtest accuracy look better than it would actually be in production.
    def _time_split(self, df, test_season):
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]
        return train_df, test_df

    def train(self, player_game_stats_df, test_season, schedules_df=None):
        features_df, feature_cols = self.feature_builder.build(player_game_stats_df, schedules_df)
        self.feature_cols = feature_cols

        train_df, test_df = self._time_split(features_df, test_season)
        logger.info(f"[{self.stat_column}] train rows: {len(train_df)}, test rows: {len(test_df)}")

        if len(train_df) < 50 or len(test_df) < 20:
            logger.warning(f"[{self.stat_column}] Not enough data to train reliably, skipping")
            return None

        X_train, y_train = train_df[feature_cols], train_df[self.stat_column]
        X_test, y_test = test_df[feature_cols], test_df[self.stat_column]

        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        residuals = y_test.values - preds
        mae = np.abs(residuals).mean()

        # Baseline: what if we just guessed each player's own rolling 3-game average
        # with no model at all? If our model can't beat this, it's not adding value.
        baseline_mae = np.abs(y_test.values - test_df["roll3"].values).mean()

        self.residual_std = residuals.std()

        logger.info(
            f"[{self.stat_column}] MAE: {mae:.1f} yds | baseline (roll3) MAE: {baseline_mae:.1f} yds | "
            f"residual std: {self.residual_std:.1f}"
        )

        return {
            "stat": self.stat_column,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "mae": mae,
            "baseline_mae": baseline_mae,
            "residual_std": self.residual_std,
        }

    # Sanity check on the confidence score itself: if we say "70% confident over"
    # across a batch of predictions, roughly 70% of those should actually go over.
    # A model that's over/under-confident is arguably worse than a coin flip,
    # because it actively misleads. This bins the test set by predicted P(over)
    # against an arbitrary line (each player's own season_avg, as a stand-in for
    # a realistic sportsbook line) and compares to actual outcome rate.
    def evaluate_calibration(self, player_game_stats_df, test_season, n_bins=5, schedules_df=None):
        from scipy.stats import norm

        features_df, feature_cols = self.feature_builder.build(player_game_stats_df, schedules_df)
        _, test_df = self._time_split(features_df, test_season)

        preds = self.model.predict(test_df[feature_cols])
        lines = test_df["season_avg"].values  # stand-in line, see docstring
        actuals = test_df[self.stat_column].values

        p_over = 1 - norm.cdf(lines, loc=preds, scale=self.residual_std)
        actual_over = (actuals > lines).astype(int)

        calib_df = pd.DataFrame({"p_over": p_over, "actual_over": actual_over})
        calib_df["bin"] = pd.qcut(calib_df["p_over"], n_bins, duplicates="drop")

        summary = calib_df.groupby("bin", observed=True).agg(
            predicted_p_over=("p_over", "mean"),
            actual_rate=("actual_over", "mean"),
            n=("actual_over", "size"),
        )
        return summary

    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"{self.stat_column}.joblib"
        joblib.dump(
            {"model": self.model, "residual_std": self.residual_std, "feature_cols": self.feature_cols},
            path,
        )
        logger.info(f"Saved model to {path}")


def main():
    import sys
    sys.path.insert(0, ".")
    import nflreadpy as nfl
    from transform.transform import PlayerStatsTransformer

    seasons = [2019, 2020, 2021, 2022, 2023, 2024]
    weekly = nfl.load_player_stats(seasons).to_pandas()
    player_stats = PlayerStatsTransformer().prepare_player_game_stats(weekly)
    schedules = nfl.load_schedules(seasons).to_pandas()

    results = []
    for stat in ["rushing_yards", "receiving_yards", "passing_yards"]:
        trainer = PropModelTrainer(stat)
        result = trainer.train(player_stats, test_season=2024, schedules_df=schedules)
        if result:
            results.append(result)
            trainer.save()
            print(f"\nCalibration check for {stat}:")
            print(trainer.evaluate_calibration(player_stats, test_season=2024, schedules_df=schedules))

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for r in results:
        beats_baseline = "beats" if r["mae"] < r["baseline_mae"] else "DOES NOT beat"
        print(f"{r['stat']:20s} MAE: {r['mae']:.1f} yds ({beats_baseline} roll3-avg baseline of {r['baseline_mae']:.1f})")


if __name__ == "__main__":
    main()
