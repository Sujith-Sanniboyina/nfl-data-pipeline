import pytest
import pandas as pd
import numpy as np
from transform.transform import TeamStatsAggregator


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


if __name__ == "__main__":
    pytest.main([__file__])