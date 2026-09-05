import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss
from pathlib import Path

from model.features import FeatureBuilder
from model.data import (
    load_player_game_stats_and_schedules,
    TEST_SEASON,
    load_pbp_data,
)
from model.probability import estimate_residual_scale, raw_p_over, apply_calibration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("model/artifacts")
EVAL_DIR = MODEL_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline(stat):
    path = MODEL_DIR / f"{stat}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def load_calibration(stat):
    path = MODEL_DIR / "calibration" / f"{stat}_calibration.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def evaluate_models(X_test, y_test, lines_test, stat):
    baseline_artifact = load_baseline(stat)
    if baseline_artifact is None:
        logger.warning(f"No baseline model for {stat}")
        return None
    model = baseline_artifact['model']
    feat_cols = baseline_artifact['feature_cols']

    X_feat = X_test[feat_cols]
    preds = model.predict(X_feat)
    mae = mean_absolute_error(y_test, preds)

    scale = estimate_residual_scale(baseline_artifact, preds)
    p_over_raw = raw_p_over(preds, lines_test, scale)
    y_binary = (y_test > lines_test).astype(int)
    acc_raw = accuracy_score(y_binary, (p_over_raw >= 0.5).astype(int))
    brier_raw = brier_score_loss(y_binary, p_over_raw)
    corr_raw = np.corrcoef(p_over_raw, y_binary)[0, 1] if len(p_over_raw) > 1 else np.nan

    cal_artifact = load_calibration(stat)
    if cal_artifact:
        p_cal = apply_calibration(p_over_raw, cal_artifact)
        acc_cal = accuracy_score(y_binary, (p_cal >= 0.5).astype(int))
        brier_cal = brier_score_loss(y_binary, p_cal)
        corr_cal = np.corrcoef(p_cal, y_binary)[0, 1] if len(p_cal) > 1 else np.nan
    else:
        acc_cal = brier_cal = corr_cal = np.nan

    return {
        'stat': stat,
        'mae': mae,
        'acc_raw': acc_raw,
        'brier_raw': brier_raw,
        'corr_raw': corr_raw,
        'acc_cal': acc_cal,
        'brier_cal': brier_cal,
        'corr_cal': corr_cal,
    }


def main():
    # Load test season (2024) data from Supabase
    df, schedules = load_player_game_stats_and_schedules()
    test_df = df[df['season'] == TEST_SEASON]

    # Load PBP for 2024 to get real weather/defensive stats
    pbp = load_pbp_data(seasons=[TEST_SEASON])

    results = []
    for stat in ["rushing_yards", "receiving_yards", "passing_yards"]:
        fb = FeatureBuilder(stat, pbp_df=pbp)
        X_test, feat_cols = fb.build(test_df, schedules)
        y_test = X_test[stat]
        lines_test = X_test['season_avg']
        X_test_feat = X_test[feat_cols]
        result = evaluate_models(X_test_feat, y_test, lines_test, stat)
        if result:
            results.append(result)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(EVAL_DIR / "evaluation_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (2024 TEST SET – WITH REAL WEATHER/DEFENSIVE FEATURES)")
    print("=" * 80)
    print(summary_df.to_string())

    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        stat = row['stat']
        print(f"\n{stat}:")
        print(f"  MAE: {row['mae']:.2f}")
        print(f"  Raw Over/Under Acc: {row['acc_raw']:.3f} (Brier {row['brier_raw']:.4f}, Corr {row['corr_raw']:.3f})")
        if not np.isnan(row['acc_cal']):
            print(f"  Calibrated Acc: {row['acc_cal']:.3f} (Brier {row['brier_cal']:.4f}, Corr {row['corr_cal']:.3f})")


if __name__ == "__main__":
    main()