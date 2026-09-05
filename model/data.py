import logging
import pandas as pd
import nflreadpy as nfl
from load.load import NFLDataLoader

logger = logging.getLogger(__name__)

DEFAULT_SEASONS = list(range(2019, 2027))   # 2019‑2025

# The base model trains on seasons < CALIBRATION_SEASON
# Residual std is fitted on CALIBRATION_SEASON (unseen to base model)
CALIBRATION_SEASON = 2024

# Final test season – untouched by training and calibration fitting
TEST_SEASON = 2025

def three_way_split(df):
    train_df = df[df["season"] < CALIBRATION_SEASON]
    calibration_df = df[df["season"] == CALIBRATION_SEASON]
    test_df = df[df["season"] == TEST_SEASON]
    return train_df, calibration_df, test_df

def train_test_split_by_season(df, test_season=TEST_SEASON):
    train_df = df[df['season'] < test_season]
    test_df = df[df['season'] == test_season]
    return train_df, test_df

def load_player_game_stats_and_schedules(seasons=None):
    seasons = list(seasons) if seasons is not None else DEFAULT_SEASONS
    loader = NFLDataLoader()
    try:
        engine = loader.get_engine()
        df = pd.read_sql("SELECT * FROM player_game_stats ORDER BY season, week, player_id", engine)
    finally:
        loader.disconnect()
    schedules = nfl.load_schedules(seasons).to_pandas()
    if "gameday" in schedules.columns:
        schedules["gameday"] = pd.to_datetime(schedules["gameday"])
    elif "game_date" in schedules.columns:
        schedules["game_date"] = pd.to_datetime(schedules["game_date"])
    else:
        logger.warning("No date column found in schedules; days_since_last_game will be 0")
    return df, schedules

def load_pbp_data(seasons=None):
    """Load play-by-play data from nflreadpy for weather/defensive features."""
    if seasons is None:
        seasons = DEFAULT_SEASONS
    pbp = nfl.load_pbp(seasons).to_pandas()
    # Keep only needed columns to save memory
    weather_cols = ["game_id", "temp", "wind", "weather"]
    available = [c for c in weather_cols if c in pbp.columns]
    if available:
        pbp = pbp[available]
    else:
        pbp = pd.DataFrame()
    return pbp