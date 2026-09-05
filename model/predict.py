import logging
from pathlib import Path

import joblib

from model.features import FeatureBuilder
from model.probability import predict_p_over

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("model/artifacts")


class PropPredictor:
    """
    Loads a trained model for one stat and turns (player, line) into a
    prediction: expected value, P(over), and a plain "Over"/"Under" call.
    """

    def __init__(self, stat_column):
        path = MODEL_DIR / f"{stat_column}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No trained model found at {path}. Run model/train.py first.")

        self.artifact = joblib.load(path)
        self.model = self.artifact["model"]
        self.feature_cols = self.artifact["feature_cols"]
        self.stat_column = stat_column
        self.feature_builder = FeatureBuilder(stat_column)

        cal_path = MODEL_DIR / "calibration" / f"{stat_column}_calibration.joblib"
        self.calibration = joblib.load(cal_path) if cal_path.exists() else None
        if self.calibration is not None:
            logger.info(f"Loaded calibration for {stat_column} (method={self.calibration['method']})")

    def predict(self, player_game_stats_df, schedules_df, player_id, season, week, team, opponent, home_game, line):
        feature_row, feature_cols = self.feature_builder.build_for_prediction(
            player_game_stats_df, schedules_df, player_id, season, week, team, opponent, home_game
        )

        # Guard against feature drift between what train.py saved and what this
        # builder produces -- fail loudly rather than silently mispredicting.
        if list(feature_cols) != list(self.feature_cols):
            raise ValueError(
                f"Feature mismatch: model was trained on {self.feature_cols}, "
                f"but current FeatureBuilder produced {feature_cols}. Retrain the model."
            )

        X = feature_row[self.feature_cols].values.reshape(1, -1)
        predicted_mean = float(self.model.predict(X)[0])

        # Same probability pipeline used by backtest.py and evaluate_v2.py:
        # per-prediction residual scale (falls back to a global constant if no
        # variance model was fit) -> Normal CDF -> calibration, if available.
        p_over = float(predict_p_over(self.artifact, predicted_mean, line, self.calibration))
        p_under = 1 - p_over

        recommendation = "Over" if p_over >= 0.5 else "Under"
        confidence = max(p_over, p_under)

        return {
            "stat": self.stat_column,
            "line": line,
            "predicted_mean": round(predicted_mean, 1),
            "p_over": round(p_over, 3),
            "p_under": round(p_under, 3),
            "recommendation": recommendation,
            "confidence": round(confidence, 3),
        }


def main():
    """
    Quick manual smoke test -- not the real API, just proves the whole path works
    end to end against live data before anything gets wired into a web endpoint.
    """
    import sys
    sys.path.insert(0, ".")
    import nflreadpy as nfl
    from transform.transform import PlayerStatsTransformer

    seasons = [2023, 2024]
    weekly = nfl.load_player_stats(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()
    player_stats = PlayerStatsTransformer().prepare_player_game_stats(weekly, schedules_df=schedules)

    # Example: Derrick Henry (BAL), predicting his week 10 2024 rushing total
    # against CIN. This game already happened (actual: 68 yards) -- using a
    # completed game as the smoke test so the result can be sanity-checked
    # against reality, not because the code requires it.
    player_id = "00-0032764"
    team = "BAL"

    predictor = PropPredictor("rushing_yards")
    result = predictor.predict(
        player_stats, schedules,
        player_id=player_id, season=2024, week=10, team=team, opponent="CIN", home_game=True,
        line=60.5,
    )
    print(f"Player: Derrick Henry ({team}) -- actual result that week was 68 rushing yards")
    print(result)


if __name__ == "__main__":
    main()