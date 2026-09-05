import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from model.features import FeatureBuilder
from model.data import load_player_game_stats_and_schedules, time_split
from model.probability import predict_p_over

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("model/artifacts")
BACKTEST_DIR = MODEL_DIR / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


class BetSimulator:
    def __init__(self, stat_column, model_artifact_path=None):
        self.stat_column = stat_column
        if model_artifact_path is None:
            model_artifact_path = MODEL_DIR / f"{stat_column}.joblib"
        if not model_artifact_path.exists():
            raise FileNotFoundError(f"Model not found: {model_artifact_path}")
        self.artifact = joblib.load(model_artifact_path)
        self.model = self.artifact["model"]
        self.feature_cols = self.artifact["feature_cols"]

        cal_path = MODEL_DIR / "calibration" / f"{stat_column}_calibration.joblib"
        self.calibration = None
        self.using_synthetic_lines = None
        if cal_path.exists():
            self.calibration = joblib.load(cal_path)
            logger.info(f"Loaded calibration for {stat_column}")

    def generate_lines(self, df):
        if 'actual_line' in df.columns:
            self.using_synthetic_lines = False
            return df['actual_line']
        else:
            logger.warning(
                f"[{self.stat_column}] No 'actual_line' column found -- generating SYNTHETIC "
                f"lines (season_avg + noise). Results below are illustrative only, not a "
                f"real profitability claim; see module docstring."
            )
            self.using_synthetic_lines = True
            noise = np.random.normal(0, 0.5 * df['season_avg'].std(), len(df))
            return df['season_avg'] + noise

    def predict_proba(self, row, line):
        X = row[self.feature_cols].values.reshape(1, -1)
        pred = self.model.predict(X)[0]
        # Same probability pipeline as predict.py and calibrate.py.
        return predict_p_over(self.artifact, pred, line, self.calibration)

    def backtest(self, df, lines=None, stake=1.0):
        if lines is None:
            lines = self.generate_lines(df)

        total_bets = 0
        correct = 0
        profit = 0.0

        for idx, row in df.iterrows():
            line = lines[idx]
            actual = row[self.stat_column]
            p_over = self.predict_proba(row, line)
            if p_over >= 0.5:
                bet = 'over'
                win = actual > line
            else:
                bet = 'under'
                win = actual < line
            total_bets += 1
            if win:
                correct += 1
                profit += stake * 0.91
            else:
                profit -= stake

        win_rate = correct / total_bets if total_bets > 0 else 0.0
        roi = profit / (total_bets * stake) if total_bets > 0 else 0.0

        logger.info(f"{self.stat_column} backtest: bets={total_bets}, win_rate={win_rate:.3f}, ROI={roi:.3f}")
        return {
            'stat': self.stat_column,
            'bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'profit': profit,
            'synthetic_lines': self.using_synthetic_lines,
        }


def main():
    df, schedules = load_player_game_stats_and_schedules()
    _, df_2024 = time_split(df, test_season=2024)

    for stat in ["rushing_yards", "receiving_yards", "passing_yards"]:
        fb = FeatureBuilder(stat)
        X, feat_cols = fb.build(df_2024, schedules)
        X = X.dropna(subset=feat_cols + [stat])

        sim = BetSimulator(stat)
        result = sim.backtest(X)
        pd.DataFrame([result]).to_csv(BACKTEST_DIR / f"{stat}_backtest.csv", index=False)
        line_note = " (SYNTHETIC lines -- illustrative only)" if result['synthetic_lines'] else ""
        print(f"{stat}: win_rate={result['win_rate']:.3f}, ROI={result['roi']:.3f}{line_note}")


if __name__ == "__main__":
    main()