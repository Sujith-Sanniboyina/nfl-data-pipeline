import sys
sys.path.insert(0, ".")

import pandas as pd
import nflreadpy as nfl
from model.predict import PropPredictor
from model.data import load_player_game_stats_and_schedules
from transform.transform import PlayerStatsTransformer

# ---- EDIT THESE ----
PLAYER_NAME = "Kyren Williams" # <<< set manually
STAT = "rushing_yards"
SEASON = 2026
WEEK = 1
LINE = 37.5
TEAM = "LA"          # <<< set this explicitly for 2026
# --------------------

# Load player stats and schedules from Supabase
df, schedules = load_player_game_stats_and_schedules()

# Find player_id from display name
player_row = df[df['player_name'] == PLAYER_NAME]
if player_row.empty:
    print(f"Player '{PLAYER_NAME}' not found in DB.")
    sys.exit(1)
player_id = player_row.iloc[0]['player_id']

# Find the game for that player in that week using the schedule and the given team
game_info = schedules[(schedules['season'] == SEASON) & (schedules['week'] == WEEK)]
if game_info.empty:
    print(f"No schedule found for season {SEASON}, week {WEEK}.")
    sys.exit(1)

# Determine opponent and home_game from the schedule
home = game_info[game_info['home_team'] == TEAM]
away = game_info[game_info['away_team'] == TEAM]
if not home.empty:
    opponent = home.iloc[0]['away_team']
    home_game = True
elif not away.empty:
    opponent = away.iloc[0]['home_team']
    home_game = False
else:
    print(f"Team {TEAM} not found in schedule for week {WEEK}")
    sys.exit(1)

# Build predictor
predictor = PropPredictor(STAT)

# Get prediction
result = predictor.predict(
    df, schedules,
    player_id=player_id,
    season=SEASON,
    week=WEEK,
    team=TEAM,
    opponent=opponent,
    home_game=home_game,
    line=LINE
)

print("\n" + "="*60)
print(f"PREDICTION for {PLAYER_NAME} ({TEAM}) – Week {WEEK} {SEASON} vs {opponent}")
print("="*60)
print(f"  Stat: {STAT}")
print(f"  Line: {LINE}")
print(f"  Expected yards: {result['predicted_mean']:.1f}")
print(f"  P(over) (calibrated): {result['p_over']:.3f}")
print(f"  P(under): {result['p_under']:.3f}")
print(f"  Recommendation: {result['recommendation']} (confidence {result['confidence']:.3f})")
print("="*60)