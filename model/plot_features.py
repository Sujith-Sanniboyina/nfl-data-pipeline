"""
Generate feature importance plots from trained XGBoost models.
"""
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from pathlib import Path

MODEL_DIR = Path("model/artifacts")
PLOT_DIR = MODEL_DIR / "feature_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def plot_feature_importance(stat_column):
    artifact_path = MODEL_DIR / f"{stat_column}.joblib"
    if not artifact_path.exists():
        print(f"Model for {stat_column} not found. Train first.")
        return

    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title(f'XGBoost Feature Importance - {stat_column}')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{stat_column}_feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved feature importance plot for {stat_column} to {PLOT_DIR / f'{stat_column}_feature_importance.png'}")

if __name__ == "__main__":
    for stat in ["rushing_yards", "receiving_yards", "passing_yards"]:
        plot_feature_importance(stat)