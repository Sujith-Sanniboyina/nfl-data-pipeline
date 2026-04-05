import sqlite3
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Loads transformed NFL data into a SQLite database
class NFLDataLoader:

    def __init__(self, db_path="data/nfl_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None

    # Opens a connection to the database if one isn't already open
    def connect(self):
        if self.connection is None:
            self.connection = sqlite3.connect(str(self.db_path))
            logger.info(f"Connected to database: {self.db_path}")

        return self.connection

    # Closes the database connection
    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")

    # Reads and runs the schema SQL file to set up the tables
    def create_schema(self, schema_path="load/schema.sql"):
        conn = self.connect()
        cursor = conn.cursor()

        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            cursor.executescript(schema_sql)
            conn.commit()
            logger.info("Database schema created successfully")

        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            conn.rollback()
            raise

    # Inserts team-season stats into the database and returns the row count
    def load_team_season_stats(self, df):
        if df.empty:
            logger.warning("No team-season data to load")
            return 0

        conn = self.connect()

        try:
            df.to_sql('team_season_stats', conn, if_exists='replace', index=False)

            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM team_season_stats")
            count = cursor.fetchone()[0]

            conn.commit()
            logger.info(f"Loaded {count} team-season records")

            return count

        except Exception as e:
            logger.error(f"Error loading team-season stats: {e}")
            conn.rollback()
            return 0

    # Inserts team-game stats into the database and returns the row count
    def load_team_game_stats(self, df):
        if df.empty:
            logger.warning("No team-game data to load")
            return 0

        conn = self.connect()

        try:
            df.to_sql('team_game_stats', conn, if_exists='replace', index=False)

            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM team_game_stats")
            count = cursor.fetchone()[0]

            conn.commit()
            logger.info(f"Loaded {count} team-game records")

            return count

        except Exception as e:
            logger.error(f"Error loading team-game stats: {e}")
            conn.rollback()
            return 0

    # Queries the top N teams for a given season ordered by win percentage
    def get_top_teams(self, season, limit=10):
        conn = self.connect()

        query = """
        SELECT season, team, wins, losses, win_percentage, total_points, total_points_allowed
        FROM team_season_stats
        WHERE season = ?
        ORDER BY win_percentage DESC, total_points DESC
        LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=(season, limit))
        return df

    # Queries all stats for a specific team and season
    def get_team_stats(self, team, season):
        conn = self.connect()

        query = """
        SELECT *
        FROM team_season_stats
        WHERE team = ? AND season = ?
        """

        df = pd.read_sql_query(query, conn, params=(team, season))
        return df


def main():
    loader = NFLDataLoader("test_nfl_data.db")

    try:
        loader.create_schema()

        test_data = pd.DataFrame({
            'season': [2023],
            'team': ['KC'],
            'total_points': [450],
            'avg_points': [26.5],
            'total_points_allowed': [350],
            'avg_points_allowed': [20.6],
            'total_yards': [5500],
            'avg_yards': [323.5],
            'total_turnovers': [15],
            'avg_turnovers': [0.9],
            'total_touchdowns': [55],
            'avg_touchdowns': [3.2],
            'total_plays': [1000],
            'avg_plays': [58.8],
            'wins': [12],
            'losses': [5],
            'win_percentage': [0.706]
        })

        loader.load_team_season_stats(test_data)

        top_teams = loader.get_top_teams(2023)
        print("Top teams for 2023:")
        print(top_teams)

    finally:
        loader.disconnect()


if __name__ == "__main__":
    main()