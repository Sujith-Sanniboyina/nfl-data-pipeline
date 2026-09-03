import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Which position group each stat is relevant for -- used to compute opponent
# defensive strength against that position specifically (a WR's receiving yards
# depend on the opponent's pass defense, not their run defense).
STAT_POSITION_GROUPS = {
    "rushing_yards": ["RB", "FB", "QB"],
    "receiving_yards": ["WR", "TE", "RB"],
    "passing_yards": ["QB"],
}


# Auxiliary "usage" columns to roll into features per stat -- these are leading
# indicators of a player's role/opportunity (not outcome), distinct from the raw
# yardage stat itself. A rising target_share is predictive going forward even if
# recent receiving_yards has been mediocre due to bad luck/drops.
STAT_USAGE_COLUMNS = {
    "rushing_yards": ["rushing_attempts"],
    "receiving_yards": ["targets", "target_share", "air_yards_share", "wopr"],
    "passing_yards": ["passing_attempts"],
}


class FeatureBuilder:
    """
    Builds model-ready features from player_game_stats. Every rolling/aggregate
    feature is computed using only games *before* the target week (shift(1) or an
    explicit prior-weeks filter) -- using same-week or future data here would leak
    the answer into the features and produce a model that looks great on paper and
    is useless in production, since none of the current-week stats exist yet at
    prediction time.
    """

    def __init__(self, stat_column):
        if stat_column not in STAT_POSITION_GROUPS:
            raise ValueError(f"Unsupported stat_column: {stat_column}")
        self.stat_column = stat_column
        self.relevant_positions = STAT_POSITION_GROUPS[stat_column]

    # Rolling averages + season-to-date average for a single player, computed
    # using only prior games (shifted by 1) so the current week's own result
    # never leaks into its own features. Also rolls the stat-specific usage
    # columns (targets, target_share, etc.) the same way.
    def _add_player_rolling_features(self, df):
        df = df.sort_values(["player_id", "season", "week"]).copy()
        grouped = df.groupby(["player_id", "season"])

        for col in [self.stat_column] + STAT_USAGE_COLUMNS.get(self.stat_column, []):
            prefix = "usage_" if col != self.stat_column else ""
            series = grouped[col]
            df[f"{prefix}roll3_{col}" if prefix else "roll3"] = series.transform(
                lambda s: s.shift(1).rolling(3, min_periods=1).mean()
            )
            if col == self.stat_column:
                df["roll5"] = series.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
                df["season_avg"] = series.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

        df["games_played"] = grouped.cumcount()  # games played *before* this one, this season

        return df

    # For each team-week, the average of `stat_column` that opponent has allowed
    # to the relevant position group so far *this season* (prior weeks only).
    # This is what lets the model account for "tough matchup vs. a weak run
    # defense" instead of only looking at the player's own recent form.
    def _add_opponent_defense_features(self, df, full_league_df):
        position_rows = full_league_df[full_league_df["position"].isin(self.relevant_positions)].copy()
        position_rows = position_rows.sort_values(["opponent", "season", "week"])

        # allowed_by_team_week: total stat allowed by (opponent, season, week)
        allowed = (
            position_rows.groupby(["opponent", "season", "week"])[self.stat_column]
            .sum()
            .reset_index()
            .rename(columns={"opponent": "team", self.stat_column: "allowed_that_week"})
        )
        allowed = allowed.sort_values(["team", "season", "week"])

        # Prior-weeks-only rolling average of yards allowed, per team per season.
        allowed["opp_def_avg_allowed"] = (
            allowed.groupby(["team", "season"])["allowed_that_week"]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )

        df = df.merge(
            allowed[["team", "season", "week", "opp_def_avg_allowed"]],
            left_on=["opponent", "season", "week"],
            right_on=["team", "season", "week"],
            how="left",
            suffixes=("", "_opp"),
        )
        df = df.drop(columns=["team_opp"], errors="ignore")

        return df

    # Vegas spread/total (known pre-game -- no leakage) converted into each team's
    # implied point total. This is a proxy for game script: heavily favored teams
    # tend to run more (protecting a lead), trailing teams tend to pass more
    # (catching up), and a high game total means more overall yardage to go around.
    # schedules_df should come from nflreadpy's load_schedules().
    def add_vegas_features(self, df, schedules_df):
        home = schedules_df[["season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
        home["implied_team_total"] = home["total_line"] / 2 - home["spread_line"] / 2
        home = home.rename(columns={"home_team": "team"}).drop(columns=["away_team"])

        away = schedules_df[["season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
        away["implied_team_total"] = away["total_line"] / 2 + away["spread_line"] / 2
        away = away.rename(columns={"away_team": "team"}).drop(columns=["home_team"])

        implied = pd.concat([
            home[["season", "week", "team", "implied_team_total"]],
            away[["season", "week", "team", "implied_team_total"]],
        ])

        df = df.merge(implied, on=["season", "week", "team"], how="left")
        league_avg_implied = implied["implied_team_total"].mean()
        df["implied_team_total"] = df["implied_team_total"].fillna(league_avg_implied)

        return df

    # Shared feature-computation pipeline used by both build() (training, drops
    # rows missing the target) and build_for_prediction() (inference, target is
    # unknown/future so it can't be dropped on).
    def _compute_features(self, df, schedules_df=None):
        relevant = df[df["position"].isin(self.relevant_positions)].copy()
        relevant = self._add_player_rolling_features(relevant)
        relevant = self._add_opponent_defense_features(relevant, df)

        relevant["home_game"] = relevant["home_game"].fillna(False).astype(int)

        usage_feature_cols = [
            f"usage_roll3_{col}" for col in STAT_USAGE_COLUMNS.get(self.stat_column, [])
        ]
        feature_cols = ["roll3", "roll5", "season_avg", "games_played", "home_game", "opp_def_avg_allowed"] + usage_feature_cols

        if schedules_df is not None:
            relevant = self.add_vegas_features(relevant, schedules_df)
            feature_cols.append("implied_team_total")

        relevant = relevant[relevant["games_played"] >= 1].copy()
        relevant["opp_def_avg_allowed"] = relevant["opp_def_avg_allowed"].fillna(
            relevant[self.stat_column].mean()
        )
        for col in usage_feature_cols:
            relevant[col] = relevant[col].fillna(0)

        return relevant, feature_cols

    # Builds the full feature set + target for one stat, ready for model training.
    # Drops early-season rows with no rolling history (games_played == 0), since
    # there's no signal yet to predict from at that point.
    def build(self, player_game_stats_df, schedules_df=None):
        relevant, feature_cols = self._compute_features(player_game_stats_df, schedules_df)
        relevant = relevant.dropna(subset=feature_cols + [self.stat_column])

        logger.info(f"Built {len(relevant)} feature rows for {self.stat_column}")

        return relevant, feature_cols

    # Builds a single feature row for a player's *next*, not-yet-played game --
    # used at prediction time, when the target stat obviously doesn't exist yet.
    # Works by appending one synthetic "future" row (known season/week/team/
    # opponent/home_game, stat value NaN) onto the player's real history, then
    # running the exact same rolling-feature pipeline used in training. Because
    # every rolling feature is shift(1)'d, the synthetic row's own NaN stat is
    # never read -- it only ever contributes its covariates (who/where/when).
    def build_for_prediction(self, player_game_stats_df, schedules_df, player_id, season, week, team, opponent, home_game):
        player_rows = player_game_stats_df[player_game_stats_df["player_id"] == player_id]
        if player_rows.empty:
            raise ValueError(f"No historical data found for player_id={player_id}")

        template = player_rows.iloc[-1].to_dict()
        future_row = {**template}
        future_row.update({
            "season": season, "week": week, "team": team, "opponent": opponent,
            "home_game": home_game, self.stat_column: None,
        })
        for col in STAT_USAGE_COLUMNS.get(self.stat_column, []):
            future_row[col] = None

        df_with_future = pd.concat(
            [player_game_stats_df, pd.DataFrame([future_row])], ignore_index=True
        )

        relevant, feature_cols = self._compute_features(df_with_future, schedules_df)
        future_features = relevant[
            (relevant["player_id"] == player_id) & (relevant["season"] == season) & (relevant["week"] == week)
        ]
        if future_features.empty:
            raise ValueError("Could not build features for the requested game (check player has prior games this season)")

        return future_features.iloc[0][feature_cols], feature_cols
