import logging
from pathlib import Path
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import optuna
from sklearn.model_selection import cross_val_score

from model.features import FeatureBuilder
from model.data import (
    load_player_game_stats_and_schedules,
    three_way_split,
    CALIBRATION_SEASON,
    TEST_SEASON,
    load_pbp_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("model/artifacts")


class PropModelTrainer:

    def __init__(self, stat_column, pbp_df=None):
        self.stat_column = stat_column
        self.pbp_df = pbp_df
        self.feature_builder = FeatureBuilder(stat_column, pbp_df=pbp_df)
        self.model = None
        self.residual_std = None
        self.residual_std_model = None
        self.feature_cols = None

    def _fit_residual_std_model(self, preds, residuals):
        if len(preds) < 30:
            logger.info(f"[{self.stat_column}] Not enough holdout rows to fit a variance model, using global residual_std")
            return None

        abs_resid = np.abs(residuals)
        variance_model = LinearRegression()
        variance_model.fit(preds.reshape(-1, 1), abs_resid)

        check_preds = np.linspace(preds.min(), preds.max(), 20).reshape(-1, 1)
        if (variance_model.predict(check_preds) <= 0).any():
            logger.warning(f"[{self.stat_column}] Variance model degenerate, falling back to global residual_std")
            return None

        return variance_model

    def tune_hyperparameters(self, X_train, y_train, n_trials=30):
        """Use Optuna to find best hyperparameters on training data."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': 42,
            }
            model = XGBRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
            return -scores.mean()  # Optuna minimizes

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        logger.info(f"Best hyperparameters for {self.stat_column}: {best_params}")
        return best_params

    def train(self, player_game_stats_df, schedules_df=None, tune=False):
        features_df, feature_cols = self.feature_builder.build(player_game_stats_df, schedules_df)
        self.feature_cols = feature_cols

        train_df, calib_df, test_df = three_way_split(features_df)
        logger.info(
            f"[{self.stat_column}] train rows: {len(train_df)} (< {CALIBRATION_SEASON}), "
            f"calibration-holdout rows: {len(calib_df)} ({CALIBRATION_SEASON}), "
            f"test rows: {len(test_df)} ({TEST_SEASON})"
        )

        if len(train_df) < 50 or len(calib_df) < 20 or len(test_df) < 20:
            logger.warning(f"[{self.stat_column}] Not enough data to train reliably, skipping")
            return None

        X_train, y_train = train_df[feature_cols], train_df[self.stat_column]
        X_calib, y_calib = calib_df[feature_cols], calib_df[self.stat_column]
        X_test, y_test = test_df[feature_cols], test_df[self.stat_column]

        # Hyperparameter tuning if requested
        if tune:
            best_params = self.tune_hyperparameters(X_train, y_train)
        else:
            best_params = {
                'n_estimators': 200,
                'max_depth': 3,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
            }

        self.model = XGBRegressor(**best_params)
        self.model.fit(X_train, y_train)

        # Fit residual std on calibration holdout
        preds_calib = self.model.predict(X_calib)
        residuals_calib = y_calib.values - preds_calib
        self.residual_std = residuals_calib.std()
        self.residual_std_model = self._fit_residual_std_model(preds_calib, residuals_calib)

        # Report on test season
        preds_test = self.model.predict(X_test)
        mae = np.abs(y_test.values - preds_test).mean()
        baseline_mae = np.abs(y_test.values - test_df["roll5"].values).mean()

        logger.info(
            f"[{self.stat_column}] Test MAE: {mae:.1f} yds | baseline (roll5) MAE: {baseline_mae:.1f} yds | "
            f"residual std (calib holdout): {self.residual_std:.1f} | "
            f"variance model: {'fitted' if self.residual_std_model is not None else 'fallback to constant'}"
        )

        return {
            "stat": self.stat_column,
            "train_rows": len(train_df),
            "calib_rows": len(calib_df),
            "test_rows": len(test_df),
            "mae": mae,
            "baseline_mae": baseline_mae,
            "residual_std": self.residual_std,
        }

    def adversarial_validation(self, player_game_stats_df, schedules_df=None):
        # (unchanged – keep the same as before)
        features_df, feature_cols = self.feature_builder.build(player_game_stats_df, schedules_df)
        self.feature_cols = feature_cols

        season_results = {}
        for test_season in [CALIBRATION_SEASON, TEST_SEASON]:
            train_df = features_df[features_df["season"] < test_season]
            test_df = features_df[features_df["season"] == test_season]
            if len(train_df) < 50 or len(test_df) < 20:
                continue
            X_train = train_df[feature_cols]
            y_train = train_df[self.stat_column]
            X_test = test_df[feature_cols]
            y_test = test_df[self.stat_column]

            model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            season_results[test_season] = mae

        players = features_df['player_id'].unique()
        np.random.seed(42)
        sample_players = np.random.choice(players, size=min(10, len(players)), replace=False)
        player_results = []
        for player in sample_players:
            train_df = features_df[features_df['player_id'] != player]
            test_df = features_df[features_df['player_id'] == player]
            if len(train_df) < 50 or len(test_df) < 5:
                continue
            X_train = train_df[feature_cols]
            y_train = train_df[self.stat_column]
            X_test = test_df[feature_cols]
            y_test = test_df[self.stat_column]

            model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            player_results.append(mae)

        train_full = features_df[features_df["season"] < CALIBRATION_SEASON]
        X_full = train_full[feature_cols]
        y_full = train_full[self.stat_column]
        in_sample_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42)
        in_sample_model.fit(X_full, y_full)
        preds_full = in_sample_model.predict(X_full)
        mae_in_sample = mean_absolute_error(y_full, preds_full)

        avg_player_mae = np.mean(player_results) if player_results else None

        logger.info(f"Adversarial validation for {self.stat_column}:")
        logger.info(f"  In-sample MAE: {mae_in_sample:.2f}")
        logger.info(f"  Season split MAEs: {season_results}")
        logger.info(f"  Average player-holdout MAE: {avg_player_mae:.2f}")

        return {
            'in_sample_mae': mae_in_sample,
            'season_split_maes': season_results,
            'player_holdout_mae_avg': avg_player_mae,
        }

    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"{self.stat_column}.joblib"
        joblib.dump(
            {
                "model": self.model,
                "residual_std": self.residual_std,
                "residual_std_model": self.residual_std_model,
                "feature_cols": self.feature_cols,
            },
            path,
        )
        logger.info(f"Saved model to {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    args = parser.parse_args()

    # Load player stats and schedules from Supabase
    df, schedules = load_player_game_stats_and_schedules()
    # Load play-by-play for weather/defensive features (seasons used in training: 2019-2022)
    pbp = load_pbp_data(seasons=list(range(2019, 2023)))  # only for training seasons to avoid leakage
    # Also include 2023? Actually we only need features for training; we'll use pbp for all years up to 2022.
    # For calibration and test, we'll also need pbp, but we can load later. We'll pass pbp to FeatureBuilder.

    results = []
    for stat in ["rushing_yards", "receiving_yards", "passing_yards"]:
        trainer = PropModelTrainer(stat, pbp_df=pbp)
        result = trainer.train(df, schedules_df=schedules, tune=args.tune)
        if result:
            results.append(result)
            trainer.save()
            adv = trainer.adversarial_validation(df, schedules)
            print(f"\nAdversarial validation for {stat}: {adv}")

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for r in results:
        beats_baseline = "beats" if r["mae"] < r["baseline_mae"] else "DOES NOT beat"
        print(f"{r['stat']:20s} MAE: {r['mae']:.1f} yds ({beats_baseline} roll5-avg baseline of {r['baseline_mae']:.1f})")


if __name__ == "__main__":
    main()