# api.py
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from model.data import load_player_game_stats_and_schedules
from model.predict import PropPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data once at startup (this is relatively fast)
logger.info("Loading player stats and schedules from Supabase...")
df, schedules = load_player_game_stats_and_schedules()
logger.info(f"Loaded {len(df)} player-game rows")

app = FastAPI(title="PropEdge API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy‑loaded predictors (only load when needed)
_predictors = {}

def get_predictor(stat: str):
    if stat not in _predictors:
        logger.info(f"Loading predictor for {stat}...")
        _predictors[stat] = PropPredictor(stat)
    return _predictors[stat]

class PredictRequest(BaseModel):
    player: str
    propType: str
    line: float
    team: str
    opponent: str
    season: int = 2026
    week: int = 1

@app.get("/players")
async def get_players():
    """Return a list of all players with name, team, position for autocomplete."""
    players = df[["player_name", "team", "position"]].drop_duplicates(subset=["player_name"])
    return players.to_dict(orient="records")

@app.post("/predict")
async def predict(request: PredictRequest):
    stat_map = {
        "rushing": "rushing_yards",
        "receiving": "receiving_yards",
        "passing": "passing_yards",
    }
    if request.propType not in stat_map:
        raise HTTPException(status_code=400, detail="Invalid propType")
    stat = stat_map[request.propType]

    # Case‑insensitive lookup
    player_rows = df[df["player_name"].str.lower() == request.player.lower()]
    if player_rows.empty:
        raise HTTPException(status_code=404, detail=f"Player '{request.player}' not found")
    player_id = player_rows.iloc[0]["player_id"]

    # Get the predictor (lazy‑loaded)
    predictor = get_predictor(stat)

    result = predictor.predict(
        player_game_stats_df=df,
        schedules_df=schedules,
        player_id=player_id,
        season=request.season,
        week=request.week,
        team=request.team,
        opponent=request.opponent,
        home_game=False,          # UI can be extended later
        line=request.line
    )

    confidence = max(result["p_over"], result["p_under"]) * 100

    return {
        "predicted": result["predicted_mean"],
        "confidence": round(confidence, 1),
        "p_over": result["p_over"],
        "recommendation": result["recommendation"],
        "line": result["line"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)