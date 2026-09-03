import pytest
import pandas as pd
import shutil
from pathlib import Path
from sqlalchemy import text
from load.load import NFLDataLoader


class TestNFLDataLoader:

    def setup_method(self):
        self.test_db = "test_data/test_nfl.db"
        self.loader = NFLDataLoader(db_path=self.test_db)

    def teardown_method(self):
        self.loader.disconnect()
        if Path("test_data").exists():
            shutil.rmtree("test_data")

    def test_connect_creates_engine(self):
        engine = self.loader.connect()
        assert engine is not None
        assert self.loader.engine is not None

    def test_create_schema(self):
        self.loader.connect()
        self.loader.create_schema()

        with self.loader.engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            tables = [row[0] for row in result]

        assert 'team_season_stats' in tables
        assert 'team_game_stats' in tables
        assert 'player_game_stats' in tables

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

    def test_load_replaces_rows_without_dropping_table(self):
        # Regression test: load methods used to call to_sql(if_exists='replace'),
        # which drops the table and rebuilds it from the DataFrame, silently losing
        # the UNIQUE constraint and indexes from schema.sql. Loading twice should
        # leave the table (and its constraints) intact, just with fresh row contents.
        first_batch = pd.DataFrame({
            'season': [2023], 'team': ['KC'], 'total_points': [450], 'avg_points': [26.5],
            'total_points_allowed': [350], 'avg_points_allowed': [20.6], 'total_yards': [5500],
            'avg_yards': [323.5], 'total_turnovers': [15], 'avg_turnovers': [0.9],
            'total_touchdowns': [55], 'avg_touchdowns': [3.2], 'total_plays': [1000],
            'avg_plays': [58.8], 'wins': [12], 'losses': [5], 'win_percentage': [0.706]
        })
        second_batch = pd.DataFrame({
            'season': [2024], 'team': ['SF'], 'total_points': [400], 'avg_points': [23.5],
            'total_points_allowed': [300], 'avg_points_allowed': [17.6], 'total_yards': [5200],
            'avg_yards': [305.9], 'total_turnovers': [10], 'avg_turnovers': [0.6],
            'total_touchdowns': [48], 'avg_touchdowns': [2.8], 'total_plays': [950],
            'avg_plays': [55.9], 'wins': [11], 'losses': [6], 'win_percentage': [0.647]
        })

        self.loader.connect()
        self.loader.create_schema()
        self.loader.load_team_season_stats(first_batch)
        self.loader.load_team_season_stats(second_batch)

        with self.loader.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM team_season_stats"))
            count = result.scalar()
            # The UNIQUE(season, team) index should still exist -- proves the table
            # wasn't dropped and recreated by the second load.
            idx_result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='team_season_stats';"
            ))
            index_names = [row[0] for row in idx_result]

        assert count == 1  # only the second batch's row -- first was replaced, not appended
        assert len(index_names) > 0

    def test_load_player_game_stats(self):
        test_data = pd.DataFrame({
            'player_id': ['00-1234'],
            'player_name': ['Test Player'],
            'position': ['RB'],
            'team': ['KC'],
            'opponent': ['SF'],
            'season': [2024],
            'week': [1],
            'home_game': [True],
            'rushing_yards': [85],
            'rushing_attempts': [18],
            'receiving_yards': [20],
            'receptions': [2],
            'targets': [3],
            'passing_yards': [0],
            'passing_attempts': [0],
            'rushing_tds': [1],
            'receiving_tds': [0],
            'passing_tds': [0],
        })

        self.loader.connect()
        self.loader.create_schema()
        count = self.loader.load_player_game_stats(test_data)

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
