import logging
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold
import joblib
from pathlib import Path

from model.features import FeatureBuilder
from model.data import (
    load_player_game_stats_and_schedules,
    train_test_split_by_season,
    TEST_SEASON,
    load_pbp_data,
)
from model.probability import estimate_residual_scale, raw_p_over, apply_calibration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("model/artifacts")
CALIB_DIR = MODEL_DIR / "calibration"
CALIB_DIR.mkdir(parents=True, exist_ok=True)


class Calibrator:
    def __init__(self, stat_column, base_model_path=None, cv_folds=5):
        self.stat_column = stat_column
        if base_model_path is None:
            base_model_path = MODEL_DIR / f"{stat_column}.joblib"
        if not base_model_path.exists():
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
        self.base_artifact = joblib.load(base_model_path)
        self.base_model = self.base_artifact["model"]
        self.feature_cols = self.base_artifact["feature_cols"]
        self.cv_folds = cv_folds
        self.calibration_model = None
        self.method = None

    def _get_out_of_fold_predictions(self, X, y, lines):
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        oof_p_over = np.zeros(len(X))
        oof_y_binary = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            lines_train_fold = lines.iloc[train_idx]

            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            lines_val_fold = lines.iloc[val_idx]

            from xgboost import XGBRegressor
            temp_model = XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            )
            temp_model.fit(X_train_fold[self.feature_cols], y_train_fold)

            raw_preds_val = temp_model.predict(X_val_fold[self.feature_cols])
            scale_val = estimate_residual_scale(self.base_artifact, raw_preds_val)
            p_over_val = raw_p_over(raw_preds_val, lines_val_fold, scale_val)

            oof_p_over[val_idx] = p_over_val
            oof_y_binary[val_idx] = (y_val_fold > lines_val_fold).astype(int)

        return oof_p_over, oof_y_binary

    def calibrate(self, X_train, y_train, lines_train, X_test, y_test, lines_test):
        logger.info(f"[{self.stat_column}] Generating out‑of‑fold predictions for calibration (CV={self.cv_folds})...")
        oof_p_over, oof_y_binary = self._get_out_of_fold_predictions(X_train, y_train, lines_train)
        oof_p_over = np.clip(oof_p_over, 1e-6, 1 - 1e-6)

        # Cross‑validate calibration methods
        from sklearn.model_selection import cross_val_score

        def platt_cv(X, y):
            lr = LogisticRegression()
            scores = cross_val_score(lr, X, y, cv=min(3, self.cv_folds), scoring='neg_brier_score')
            return -scores.mean()

        def isotonic_cv(X, y):
            kf = KFold(n_splits=min(3, self.cv_folds), shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(X_tr, y_tr)
                preds = iso.predict(X_val)
                scores.append(brier_score_loss(y_val, preds))
            return np.mean(scores)

        def temperature_scale(p, T):
            logit = np.log(p / (1 - p))
            return expit(logit / T)

        def temperature_loss(T, p, y):
            p_cal = temperature_scale(p, T)
            return brier_score_loss(y, p_cal)

        def temperature_cv(X, y):
            best_score = float('inf')
            for T in np.linspace(0.1, 10, 50):
                score = 0
                kf = KFold(n_splits=min(3, self.cv_folds), shuffle=True, random_state=42)
                for train_idx, val_idx in kf.split(X):
                    X_tr, y_tr = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]
                    res = minimize(lambda t: temperature_loss(t, X_tr, y_tr), x0=1.0, bounds=[(0.1, 10.0)])
                    T_opt = res.x[0]
                    p_cal = temperature_scale(X_val, T_opt)
                    score += brier_score_loss(y_val, p_cal)
                score /= kf.get_n_splits()
                if score < best_score:
                    best_score = score
            return best_score

        X_cv = oof_p_over.reshape(-1, 1)
        y_cv = oof_y_binary

        platt_score = platt_cv(X_cv, y_cv)
        isotonic_score = isotonic_cv(X_cv, y_cv)
        temperature_score = temperature_cv(X_cv, y_cv)

        logger.info(f"CV Brier scores (on OOF) - Platt: {platt_score:.4f}, Isotonic: {isotonic_score:.4f}, Temp: {temperature_score:.4f}")

        methods = {'platt': platt_score, 'isotonic': isotonic_score, 'temperature': temperature_score}
        best_method = min(methods, key=methods.get)
        logger.info(f"Best method: {best_method}")

        if best_method == 'platt':
            cal_model = LogisticRegression()
            cal_model.fit(X_cv, y_cv)
        elif best_method == 'isotonic':
            cal_model = IsotonicRegression(out_of_bounds='clip')
            cal_model.fit(X_cv.ravel(), y_cv)
        else:
            res = minimize(lambda t: temperature_loss(t, X_cv.ravel(), y_cv),
                           x0=1.0, bounds=[(0.1, 10.0)])
            cal_model = res.x[0]

        self.calibration_model = cal_model
        self.method = best_method

        # Evaluate on test set (2024)
        raw_preds_test = self.base_model.predict(X_test[self.feature_cols])
        scale_test = estimate_residual_scale(self.base_artifact, raw_preds_test)
        p_over_test = raw_p_over(raw_preds_test, lines_test, scale_test)
        p_over_test = np.clip(p_over_test, 1e-6, 1 - 1e-6)
        y_binary_test = (y_test > lines_test).astype(int)

        cal_dict = {'method': best_method, 'model': cal_model}
        p_cal_test = apply_calibration(p_over_test, cal_dict)

        test_brier = brier_score_loss(y_binary_test, p_cal_test)
        logger.info(f"Test Brier score for {self.stat_column}: {test_brier:.4f}")

        cal_dict_save = {
            'method': best_method,
            'model': cal_model,
            'feature_cols': self.feature_cols,
            'test_brier': test_brier,
        }
        joblib.dump(cal_dict_save, CALIB_DIR / f"{self.stat_column}_calibration.joblib")
        logger.info(f"Saved calibration for {self.stat_column} to {CALIB_DIR / f'{self.stat_column}_calibration.joblib'}")
        return cal_dict_save


def main():
    # Load full player stats from Supabase
    df, schedules = load_player_game_stats_and_schedules()

    # Load PBP for weather/defensive features
    # Training set: seasons < 2024 (2019–2023), Test set: 2024
    train_seasons = list(range(2019, 2024))   # 2019-2023
    test_seasons = [2024]

    logger.info("Loading play‑by‑play data for weather/defensive features...")
    pbp_train = load_pbp_data(seasons=train_seasons)
    pbp_test = load_pbp_data(seasons=test_seasons)

    # Split player stats
    train_df, test_df = train_test_split_by_season(df, test_season=TEST_SEASON)

    for stat in ["rushing_yards", "receiving_yards", "passing_yards"]:
        logger.info(f"Processing {stat}...")

        # Build features with PBP data
        fb_train = FeatureBuilder(stat, pbp_df=pbp_train)
        X_train, _ = fb_train.build(train_df, schedules)

        fb_test = FeatureBuilder(stat, pbp_df=pbp_test)
        X_test, _ = fb_test.build(test_df, schedules)

        y_train = X_train[stat]
        lines_train = X_train['season_avg']
        y_test = X_test[stat]
        lines_test = X_test['season_avg']

        calibrator = Calibrator(stat, cv_folds=5)
        calibrator.calibrate(X_train, y_train, lines_train,
                             X_test, y_test, lines_test)


if __name__ == "__main__":
    main()