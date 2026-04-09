import pytest
import pandas as pd
import shutil
from pathlib import Path
from load.load import NFLDataLoader


class TestNFLDataLoader:

    def setup_method(self):
        self.test_db = "test_data/test_nfl.db"
        self.loader = NFLDataLoader(db_path=self.test_db)

    def teardown_method(self):
        self.loader.disconnect()
        if Path(self.test_db).exists():
            Path(self.test_db).unlink()
        if Path("test_data").exists():
            shutil.rmtree("test_data")

    def test_connect_creates_connection(self):
        conn = self.loader.connect()
        assert conn is not None
        assert self.loader.connection is not None

    def test_create_schema(self):
        self.loader.connect()
        self.loader.create_schema()

        cursor = self.loader.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        assert 'team_season_stats' in tables
        assert 'team_game_stats' in tables

    def test_load_team_season_stats(self):
        test_data = pd.DataFrame({
            'season':               [2023],
            'team':                 ['KC'],
            'total_points':         [450],
            'avg_points':           [26.5],
            'total_points_allowed': [350],
            'avg_points_allowed':   [20.6],
            'total_yards':          [5500],
            'avg_yards':            [323.5],
            'total_turnovers':      [15],
            'avg_turnovers':        [0.9],
            'total_touchdowns':     [55],
            'avg_touchdowns':       [3.2],
            'total_plays':          [1000],
            'avg_plays':            [58.8],
            'wins':                 [12],
            'losses':               [5],
            'win_percentage':       [0.706]
        })

        self.loader.connect()
        self.loader.create_schema()
        count = self.loader.load_team_season_stats(test_data)

        assert count == 1

    def test_get_top_teams(self):
        self.loader.connect()
        self.loader.create_schema()

        test_data = pd.DataFrame({
            'season':               [2023,   2023],
            'team':                 ['KC',   'SF'],
            'total_points':         [450,    440],
            'avg_points':           [26.5,   25.9],
            'total_points_allowed': [350,    340],
            'avg_points_allowed':   [20.6,   20.0],
            'total_yards':          [5500,   5400],
            'avg_yards':            [323.5,  317.6],
            'total_turnovers':      [15,     12],
            'avg_turnovers':        [0.9,    0.7],
            'total_touchdowns':     [55,     52],
            'avg_touchdowns':       [3.2,    3.1],
            'total_plays':          [1000,   980],
            'avg_plays':            [58.8,   57.6],
            'wins':                 [12,     11],
            'losses':               [5,      6],
            'win_percentage':       [0.706,  0.647]
        })

        self.loader.load_team_season_stats(test_data)

        top_teams = self.loader.get_top_teams(2023, limit=2)

        assert len(top_teams) == 2
        assert top_teams.iloc[0]['team'] == 'KC'


if __name__ == "__main__":
    pytest.main([__file__])