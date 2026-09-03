import pytest
import pandas as pd
import numpy as np
from transform.transform import TeamStatsAggregator, PlayerStatsTransformer


class TestTeamStatsAggregator:

    def setup_method(self):
        self.aggregator = TeamStatsAggregator()

    def test_calculate_team_stats(self):
        test_data = pd.DataFrame({
            'game_id':      ['test_game_1', 'test_game_1'],
            'posteam':      ['KC', 'KC'],
            'defteam':      ['SF', 'SF'],
            'yards_gained': [10, 20],
            'touchdown':    [0, 1],
            'interception': [0, 0],
            'fumble_lost':  [0, 0]
        })

        stats = self.aggregator.calculate_team_stats(test_data, 'KC')

        assert stats['team'] == 'KC'
        assert stats['total_plays'] == 2
        assert stats['offensive_plays'] == 2
        assert stats['total_yards'] == 30
        assert stats['touchdowns'] == 1

    def test_aggregate_game_stats_empty(self):
        empty_df = pd.DataFrame()
        result = self.aggregator.aggregate_game_stats(empty_df)
        assert len(result) == 0

    def test_aggregate_season_stats_empty(self):
        empty_df = pd.DataFrame()
        result = self.aggregator.aggregate_season_stats(empty_df)
        assert len(result) == 0


class TestPlayerStatsTransformer:

    def setup_method(self):
        self.transformer = PlayerStatsTransformer()

    def test_prepare_player_game_stats_maps_columns(self):
        weekly_df = pd.DataFrame({
            'player_id': ['00-1234'],
            'player_display_name': ['Test Player'],
            'position': ['RB'],
            'team': ['KC'],
            'opponent_team': ['SF'],
            'season': [2024],
            'week': [1],
            'season_type': ['REG'],
            'rushing_yards': [85],
            'carries': [18],
            'receiving_yards': [20],
            'receptions': [2],
            'targets': [3],
            'passing_yards': [0],
            'attempts': [0],
            'rushing_tds': [1],
            'receiving_tds': [0],
            'passing_tds': [0],
        })

        result = self.transformer.prepare_player_game_stats(weekly_df)

        assert len(result) == 1
        assert result.iloc[0]['player_name'] == 'Test Player'
        assert result.iloc[0]['team'] == 'KC'
        assert result.iloc[0]['rushing_attempts'] == 18

    def test_prepare_player_game_stats_filters_to_regular_season(self):
        weekly_df = pd.DataFrame({
            'player_id': ['00-1234', '00-5678'],
            'player_display_name': ['Player A', 'Player B'],
            'position': ['RB', 'WR'],
            'team': ['KC', 'SF'],
            'opponent_team': ['SF', 'KC'],
            'season': [2024, 2024],
            'week': [1, 1],
            'season_type': ['REG', 'POST'],
            'rushing_yards': [85, 0],
            'carries': [18, 0],
            'receiving_yards': [20, 90],
            'receptions': [2, 6],
            'targets': [3, 8],
            'passing_yards': [0, 0],
            'attempts': [0, 0],
            'rushing_tds': [1, 0],
            'receiving_tds': [0, 1],
            'passing_tds': [0, 0],
        })

        result = self.transformer.prepare_player_game_stats(weekly_df)

        assert len(result) == 1
        assert result.iloc[0]['player_name'] == 'Player A'

    def test_prepare_player_game_stats_empty(self):
        result = self.transformer.prepare_player_game_stats(pd.DataFrame())
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__])