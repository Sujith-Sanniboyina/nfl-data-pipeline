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

        if 'season_type' in pbp_df.columns:
            pbp_df = pbp_df[pbp_df['season_type'] == 'REG']

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


def main():
    print("Transform module ready. Run extract first to get data.")


if __name__ == "__main__":
    main()