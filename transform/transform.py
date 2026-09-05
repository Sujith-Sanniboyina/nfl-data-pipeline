import pandas as pd
import logging

logger = logging.getLogger(__name__)


# Aggregates play-by-play data into team-level and season-level statistics
class TeamStatsAggregator:

    def __init__(self):
        self.team_stats = {}

    # Loops through each game and builds a team-game record for both home and away teams
    def aggregate_game_stats(self, pbp_df):
        logger.info("Starting team-level aggregation...")

        if pbp_df.empty:
            logger.warning("No play-by-play data provided")
            return pd.DataFrame()

        if 'season_type' in pbp_df.columns:
            pbp_df = pbp_df[pbp_df['season_type'] == 'REG']

        if 'game_id' not in pbp_df.columns:
            logger.error("DataFrame missing 'game_id' column")
            return pd.DataFrame()

        team_games = []

        for game_id in pbp_df['game_id'].unique():
            game_df = pbp_df[pbp_df['game_id'] == game_id]

            if 'home_team' not in game_df.columns or 'away_team' not in game_df.columns:
                continue

            home_team = game_df['home_team'].iloc[0]
            away_team = game_df['away_team'].iloc[0]

            if 'total_home_score' in game_df.columns and 'total_away_score' in game_df.columns:
                home_score = game_df['total_home_score'].iloc[-1]
                away_score = game_df['total_away_score'].iloc[-1]
            else:
                home_score = None
                away_score = None

            season = game_df['season'].iloc[0] if 'season' in game_df.columns else None
            week = game_df['week'].iloc[0] if 'week' in game_df.columns else None

            home_stats = self.calculate_team_stats(game_df, home_team)
            away_stats = self.calculate_team_stats(game_df, away_team)

            home_stats.update({
                'game_id': game_id,
                'season': season,
                'week': week,
                'opponent': away_team,
                'points_scored': home_score,
                'points_allowed': away_score,
                'home_game': True
            })

            away_stats.update({
                'game_id': game_id,
                'season': season,
                'week': week,
                'opponent': home_team,
                'points_scored': away_score,
                'points_allowed': home_score,
                'home_game': False
            })

            team_games.append(home_stats)
            team_games.append(away_stats)

        team_games_df = pd.DataFrame(team_games)

        if len(team_games_df) > 0:
            logger.info(f"Aggregated {len(team_games_df)} team-game records")

        return team_games_df

    # Calculates offensive stats, defensive plays, yards, turnovers, and touchdowns for a team in one game
    def calculate_team_stats(self, game_df, team):
        offensive_plays = game_df[game_df['posteam'] == team]
        defensive_plays = game_df[game_df['defteam'] == team]

        stats = {
            'team': team,
            'total_plays': len(offensive_plays) + len(defensive_plays),
            'offensive_plays': len(offensive_plays),
            'defensive_plays': len(defensive_plays)
        }

        if 'yards_gained' in offensive_plays.columns:
            total_yards = offensive_plays['yards_gained'].sum()
            stats['total_yards'] = 0 if pd.isna(total_yards) else total_yards
        else:
            stats['total_yards'] = 0

        if 'interception' in offensive_plays.columns and 'fumble_lost' in offensive_plays.columns:
            interceptions = len(offensive_plays[offensive_plays['interception'] == 1])
            fumbles = len(offensive_plays[offensive_plays['fumble_lost'] == 1])
            stats['turnovers'] = interceptions + fumbles
        else:
            stats['turnovers'] = 0

        if 'touchdown' in offensive_plays.columns:
            stats['touchdowns'] = len(offensive_plays[offensive_plays['touchdown'] == 1])
        else:
            stats['touchdowns'] = 0

        return stats

    # Rolls up all team-game records into one row per team per season
    def aggregate_season_stats(self, team_games_df):
        if team_games_df.empty:
            logger.warning("No team-game data available for season aggregation")
            return pd.DataFrame()

        season_stats = team_games_df.groupby(['season', 'team']).agg({
            'points_scored': ['sum', 'mean'],
            'points_allowed': ['sum', 'mean'],
            'total_yards': ['sum', 'mean'],
            'turnovers': ['sum', 'mean'],
            'touchdowns': ['sum', 'mean'],
            'total_plays': ['sum', 'mean']
        }).reset_index()

        season_stats.columns = [
            'season', 'team',
            'total_points', 'avg_points',
            'total_points_allowed', 'avg_points_allowed',
            'total_yards', 'avg_yards',
            'total_turnovers', 'avg_turnovers',
            'total_touchdowns', 'avg_touchdowns',
            'total_plays', 'avg_plays'
        ]

        if 'points_scored' in team_games_df.columns and 'points_allowed' in team_games_df.columns:
            wins_df = team_games_df[team_games_df['points_scored'] > team_games_df['points_allowed']]
            losses_df = team_games_df[team_games_df['points_scored'] < team_games_df['points_allowed']]

            win_counts = wins_df.groupby(['season', 'team']).size().reset_index(name='wins')
            loss_counts = losses_df.groupby(['season', 'team']).size().reset_index(name='losses')

            season_stats = season_stats.merge(win_counts, on=['season', 'team'], how='left')
            season_stats = season_stats.merge(loss_counts, on=['season', 'team'], how='left')

            season_stats['wins'] = season_stats['wins'].fillna(0).astype(int)
            season_stats['losses'] = season_stats['losses'].fillna(0).astype(int)
            season_stats['win_percentage'] = season_stats['wins'] / (season_stats['wins'] + season_stats['losses'])

        logger.info(f"Aggregated season statistics for {len(season_stats)} team-seasons")

        return season_stats


# Prepares weekly player-level data (from nflreadpy's load_player_stats) for loading
# into player_game_stats. This feeds the prop predictor model.
class PlayerStatsTransformer:

    # Maps nflreadpy's load_player_stats() column names onto our schema's column names.
    COLUMN_MAP = {
        "player_id": "player_id",
        "player_display_name": "player_name",
        "position": "position",
        "team": "team",
        "opponent_team": "opponent",
        "season": "season",
        "week": "week",
        "rushing_yards": "rushing_yards",
        "carries": "rushing_attempts",
        "receiving_yards": "receiving_yards",
        "receptions": "receptions",
        "targets": "targets",
        "target_share": "target_share",
        "air_yards_share": "air_yards_share",
        "wopr": "wopr",
        "passing_yards": "passing_yards",
        "attempts": "passing_attempts",
        "rushing_tds": "rushing_tds",
        "receiving_tds": "receiving_tds",
        "passing_tds": "passing_tds",
    }

    def prepare_player_game_stats(self, weekly_df, schedules_df=None):
        if weekly_df.empty:
            logger.warning("No weekly player data provided")
            return pd.DataFrame()

        if "season_type" in weekly_df.columns:
            weekly_df = weekly_df[weekly_df["season_type"] == "REG"]

        # Only keep columns we actually have and know how to map.
        available_source_cols = [c for c in self.COLUMN_MAP if c in weekly_df.columns]
        missing = set(self.COLUMN_MAP) - set(available_source_cols)
        if missing:
            logger.warning(f"Weekly data missing expected columns (skipping): {missing}")

        prepared = weekly_df[available_source_cols].rename(
            columns={k: self.COLUMN_MAP[k] for k in available_source_cols}
        )

        prepared = self._add_home_game(prepared, schedules_df)

        prepared = prepared.dropna(subset=["player_id", "season", "week"])
        logger.info(f"Prepared {len(prepared)} player-game rows")

        return prepared

    def _add_home_game(self, prepared, schedules_df):
        """
        Sets home_game from the schedule (team == schedule's home_team for that
        season/week). Previously this was always left as NULL, which meant every
        downstream consumer's `.fillna(False)` turned it into a constant 0 --
        home/away was never actually a real feature. Falls back to leaving it NULL
        (old behavior) if no schedule is supplied, so this stays backward compatible.
        """
        required_cols = {"season", "week", "home_team", "away_team"}
        if schedules_df is None or not required_cols.issubset(schedules_df.columns):
            logger.warning("No usable schedules_df provided; home_game will be left null (uninformative)")
            prepared["home_game"] = None
            return prepared

        # Unpivot schedule rows (one row per game) into one row per (season, week, team)
        # with an is_home flag, so it can be merged directly onto prepared -- merging on
        # just (season, week) would fan out, since each week has many games/home_teams.
        home_side = schedules_df[["season", "week", "home_team"]].rename(columns={"home_team": "team"})
        home_side["is_home"] = True
        away_side = schedules_df[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
        away_side["is_home"] = False
        home_lookup = pd.concat([home_side, away_side], ignore_index=True).drop_duplicates(
            subset=["season", "week", "team"]
        )

        prepared = prepared.merge(home_lookup, on=["season", "week", "team"], how="left")
        prepared["home_game"] = prepared["is_home"].fillna(False)
        prepared = prepared.drop(columns=["is_home"])
        return prepared