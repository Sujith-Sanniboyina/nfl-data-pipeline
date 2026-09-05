import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

STAT_POSITION_GROUPS = {
    "rushing_yards": ["RB", "FB", "QB"],
    "receiving_yards": ["WR", "TE", "RB"],
    "passing_yards": ["QB"],
}

STAT_USAGE_COLUMNS = {
    "rushing_yards": ["rushing_attempts"],
    "receiving_yards": ["targets", "target_share", "air_yards_share", "wopr"],
    "passing_yards": ["passing_attempts"],
}

STAT_VOLUME_COLUMN = {
    "rushing_yards": "rushing_attempts",
    "receiving_yards": "targets",
    "passing_yards": "passing_attempts",
}


class FeatureBuilder:

    def __init__(self, stat_column, pbp_df=None):
        if stat_column not in STAT_POSITION_GROUPS:
            raise ValueError(f"Unsupported stat_column: {stat_column}")
        self.stat_column = stat_column
        self.relevant_positions = STAT_POSITION_GROUPS[stat_column]
        self.pbp_df = pbp_df  # optional play-by-play for weather/defensive features

    def _add_player_rolling_features(self, df):
        df = df.sort_values(["player_id", "season", "week"]).copy()
        grouped = df.groupby(["player_id", "season"])

        df["roll5"] = grouped[self.stat_column].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        df["roll10"] = grouped[self.stat_column].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).mean()
        )
        df["season_avg"] = grouped[self.stat_column].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

        usage_cols = STAT_USAGE_COLUMNS.get(self.stat_column, [])
        for col in usage_cols:
            if col in df.columns:
                df[f"usage_roll3_{col}"] = grouped[col].transform(
                    lambda s: s.shift(1).rolling(3, min_periods=1).mean()
                )

        df["games_played"] = grouped.cumcount()

        # Carryover prior-season performance into early-season weeks
        prior_season_avg = (
            df.groupby(["player_id", "season"])[self.stat_column]
            .mean()
            .reset_index()
            .rename(columns={self.stat_column: "prior_season_final_avg", "season": "prev_season"})
        )
        prior_season_avg["season"] = prior_season_avg["prev_season"] + 1
        df = df.merge(
            prior_season_avg[["player_id", "season", "prior_season_final_avg"]],
            on=["player_id", "season"],
            how="left",
        )
        for col in ["roll5", "roll10", "season_avg"]:
            df[col] = df[col].fillna(df["prior_season_final_avg"])
        df = df.drop(columns=["prior_season_final_avg"])

        return df

    def _add_opponent_defense_features(self, df, full_league_df):
        position_rows = full_league_df[full_league_df["position"].isin(self.relevant_positions)].copy()
        position_rows = position_rows.sort_values(["opponent", "season", "week"])

        volume_col = STAT_VOLUME_COLUMN.get(self.stat_column)
        has_volume = volume_col is not None and volume_col in position_rows.columns

        agg_spec = {self.stat_column: "sum"}
        if has_volume:
            agg_spec[volume_col] = "sum"

        allowed = (
            position_rows.groupby(["opponent", "season", "week"])
            .agg(agg_spec)
            .reset_index()
            .rename(columns={"opponent": "team", self.stat_column: "allowed_that_week"})
        )
        if has_volume:
            allowed = allowed.rename(columns={volume_col: "usage_allowed_that_week"})

        allowed = allowed.sort_values(["team", "season", "week"])
        grouped = allowed.groupby(["team", "season"])

        allowed["opp_def_avg_allowed"] = grouped["allowed_that_week"].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean()
        )

        merge_cols = ["team", "season", "week", "opp_def_avg_allowed"]

        if has_volume:
            rolling_yards = grouped["allowed_that_week"].transform(
                lambda s: s.shift(1).rolling(4, min_periods=1).sum()
            )
            rolling_usage = grouped["usage_allowed_that_week"].transform(
                lambda s: s.shift(1).rolling(4, min_periods=1).sum()
            )
            allowed["opp_def_eff_allowed"] = rolling_yards / rolling_usage.replace(0, np.nan)
            merge_cols.append("opp_def_eff_allowed")

        df = df.merge(
            allowed[merge_cols],
            left_on=["opponent", "season", "week"],
            right_on=["team", "season", "week"],
            how="left",
            suffixes=("", "_opp"),
        )
        df = df.drop(columns=["team_opp"], errors="ignore")
        return df

    def _add_matchup_features(self, df):
        df = df.sort_values(["player_id", "opponent", "season", "week"]).copy()
        df["player_vs_opponent_avg"] = (
            df.groupby(["player_id", "opponent"])[self.stat_column]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )
        return df

    def _add_game_context_features(self, df, schedules_df):
        if schedules_df is None:
            logger.warning("No schedules_df provided, skipping game context features")
            df["days_since_last_game"] = 0
            df["is_primetime"] = 0
            df["is_division_game"] = 0
            return df

        date_col = None
        for col in ['game_date', 'gameday', 'date']:
            if col in schedules_df.columns:
                date_col = col
                break

        if date_col is None:
            logger.warning("No date column found in schedules_df; setting days_since_last_game = 0")
            df["days_since_last_game"] = 0
        else:
            schedules_df[date_col] = pd.to_datetime(schedules_df[date_col])
            home = schedules_df[["season", "week", "home_team", "away_team", date_col]].copy()
            home = home.rename(columns={"home_team": "team", date_col: "game_date"})
            away = schedules_df[["season", "week", "home_team", "away_team", date_col]].copy()
            away = away.rename(columns={"away_team": "team", date_col: "game_date"})
            game_dates = pd.concat([home, away], ignore_index=True)
            game_dates = game_dates.drop_duplicates(subset=["season", "week", "team"])
            df = df.merge(game_dates, on=["season", "week", "team"], how="left")
            df = df.sort_values(["player_id", "season", "week"]).copy()
            df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
            df["days_since_last_game"] = (df["game_date"] - df["prev_game_date"]).dt.days
            df["days_since_last_game"] = df["days_since_last_game"].fillna(0)
            df = df.drop(columns=["game_date", "prev_game_date"], errors="ignore")

        if "primetime" in schedules_df.columns:
            pt = schedules_df[["season", "week", "home_team", "away_team", "primetime"]].copy()
            home_pt = pt[["season", "week", "home_team", "primetime"]].rename(columns={"home_team": "team"})
            away_pt = pt[["season", "week", "away_team", "primetime"]].rename(columns={"away_team": "team"})
            pt_all = pd.concat([home_pt, away_pt], ignore_index=True)
            df = df.merge(pt_all, on=["season", "week", "team"], how="left")
            df["is_primetime"] = df["primetime"].fillna(False).astype(int)
            df = df.drop(columns=["primetime"], errors="ignore")
        else:
            df["is_primetime"] = 0

        if "division_game" in schedules_df.columns:
            div = schedules_df[["season", "week", "home_team", "away_team", "division_game"]].copy()
            home_div = div[["season", "week", "home_team", "division_game"]].rename(columns={"home_team": "team"})
            away_div = div[["season", "week", "away_team", "division_game"]].rename(columns={"away_team": "team"})
            div_all = pd.concat([home_div, away_div], ignore_index=True)
            df = df.merge(div_all, on=["season", "week", "team"], how="left")
            df["is_division_game"] = df["division_game"].fillna(False).astype(int)
            df = df.drop(columns=["division_game"], errors="ignore")
        else:
            df["is_division_game"] = 0

        return df

    def add_vegas_features(self, df, schedules_df):
        if schedules_df is None:
            return df
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

    # ---------- NEW WEATHER & DEFENSIVE EFFICIENCY ----------
    def _add_weather_features(self, df, schedules_df):
        if self.pbp_df is None or self.pbp_df.empty:
            logger.warning("No pbp_df provided; weather features will be zero.")
            df["temp"] = 0.0
            df["wind_speed"] = 0.0
            df["precip"] = 0.0
            return df

        # Extract weather per game (first row) from pbp
        weather_cols = ["game_id", "temp", "wind", "weather"]
        available = [c for c in weather_cols if c in self.pbp_df.columns]
        if not available:
            logger.warning("Weather columns not found in pbp_df; setting to zero.")
            df["temp"] = 0.0
            df["wind_speed"] = 0.0
            df["precip"] = 0.0
            return df

        # Aggregate weather per game (take first non‑null)
        game_weather = self.pbp_df.groupby("game_id").agg({
            "temp": "first",
            "wind": "first",
            "weather": "first"
        }).reset_index()

        # Create mapping from (season, week, team) to game_id using schedules
        if schedules_df is None:
            logger.warning("No schedules_df; cannot merge weather by game.")
            df["temp"] = 0.0
            df["wind_speed"] = 0.0
            df["precip"] = 0.0
            return df

        home = schedules_df[["season", "week", "home_team", "game_id"]].rename(columns={"home_team": "team"})
        away = schedules_df[["season", "week", "away_team", "game_id"]].rename(columns={"away_team": "team"})
        game_lookup = pd.concat([home, away], ignore_index=True).drop_duplicates(subset=["season", "week", "team"])
        df = df.merge(game_lookup, on=["season", "week", "team"], how="left")
        df = df.merge(game_weather, on="game_id", how="left")

        df["temp"] = df["temp"].fillna(0).astype(float)
        df["wind_speed"] = df["wind"].fillna(0).astype(float)
        df["precip"] = df["weather"].str.contains("rain|snow|precip", case=False, na=False).astype(int) if "weather" in df.columns else 0

        df = df.drop(columns=["game_id", "weather", "wind"], errors="ignore")
        return df

    def _add_opponent_defensive_efficiency(self, df, schedules_df):
        # For now, we use the existing opp_def_avg_allowed and opp_def_eff_allowed
        # but we can add additional metrics like yards per play allowed.
        # We'll just add dummy columns to keep consistent; you can expand.
        df["opp_def_ypa"] = 0.0
        df["opp_def_ppd"] = 0.0
        return df
    # ---------------------------------------------------------

    def _compute_features(self, df, schedules_df=None):
        relevant = df[df["position"].isin(self.relevant_positions)].copy()
        relevant = self._add_player_rolling_features(relevant)
        relevant = self._add_opponent_defense_features(relevant, df)
        relevant = self._add_matchup_features(relevant)

        if schedules_df is not None:
            relevant = self._add_game_context_features(relevant, schedules_df)
            relevant = self.add_vegas_features(relevant, schedules_df)
            # Add new features after context
            relevant = self._add_weather_features(relevant, schedules_df)
            relevant = self._add_opponent_defensive_efficiency(relevant, schedules_df)

        relevant["home_game"] = relevant["home_game"].fillna(False).astype(int)

        usage_cols = STAT_USAGE_COLUMNS.get(self.stat_column, [])
        usage_feature_cols = []
        for col in usage_cols:
            fname = f"usage_roll3_{col}"
            if fname in relevant.columns:
                usage_feature_cols.append(fname)

        # Base features
        feature_cols = [
            "roll5", "roll10", "season_avg", "games_played", "home_game",
            "opp_def_avg_allowed", "player_vs_opponent_avg"
        ] + usage_feature_cols

        if "opp_def_eff_allowed" in relevant.columns:
            feature_cols.append("opp_def_eff_allowed")

        if schedules_df is not None:
            context_cols = ["days_since_last_game", "is_primetime", "is_division_game", "implied_team_total"]
            for col in context_cols:
                if col in relevant.columns:
                    feature_cols.append(col)

        # Add new weather/defensive columns if they exist
        new_cols = ["temp", "wind_speed", "precip", "opp_def_ypa", "opp_def_ppd"]
        for col in new_cols:
            if col in relevant.columns:
                feature_cols.append(col)

        # Fill missing values
        relevant["opp_def_avg_allowed"] = relevant["opp_def_avg_allowed"].fillna(relevant[self.stat_column].mean())
        if "opp_def_eff_allowed" in relevant.columns:
            relevant["opp_def_eff_allowed"] = relevant["opp_def_eff_allowed"].fillna(
                relevant["opp_def_eff_allowed"].mean()
            )
        relevant["player_vs_opponent_avg"] = relevant["player_vs_opponent_avg"].fillna(relevant["season_avg"])
        for col in usage_feature_cols:
            relevant[col] = relevant[col].fillna(0)
        if "days_since_last_game" in relevant.columns:
            relevant["days_since_last_game"] = relevant["days_since_last_game"].fillna(0)

        return relevant, feature_cols

    def build(self, player_game_stats_df, schedules_df=None):
        relevant, feature_cols = self._compute_features(player_game_stats_df, schedules_df)
        relevant = relevant.dropna(subset=feature_cols + [self.stat_column])
        logger.info(f"Built {len(relevant)} feature rows for {self.stat_column}")
        return relevant, feature_cols

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
        usage_cols = STAT_USAGE_COLUMNS.get(self.stat_column, [])
        for col in usage_cols:
            if col in future_row:
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