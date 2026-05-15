import os
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


# Loads transformed NFL data into a database (PostgreSQL or SQLite)
class NFLDataLoader:

    def __init__(self, db_path=None):
        if db_path is None:
            load_dotenv()
            self.database_url = f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
            self.is_postgresql = True
        else:
            self.database_url = f"sqlite:///{db_path}"
            self.is_postgresql = False
        
        self.engine = None
        logger.info(f"Database URL configured: {self.database_url}")

    # Get or create database engine
    def get_engine(self):
        if self.engine is None:
            self.engine = create_engine(self.database_url)
            if self.is_postgresql:
                logger.info(f"Connected to PostgreSQL database")
            else:
                logger.info(f"Connected to SQLite database")
        return self.engine

    # Close database connection
    def disconnect(self):
        if self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Database connection closed")

    # For backward compatibility with main.py
    def connect(self):
        return self.get_engine()

    # Reads and runs the schema SQL file to set up the tables
    def create_schema(self, schema_path="load/schema.sql"):
        engine = self.get_engine()

        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            with engine.connect() as conn:
                statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
                for statement in statements:
                    if statement:
                        conn.execute(text(statement))
                conn.commit()

            logger.info("Database schema created successfully")

        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            raise

    # Inserts team-season stats into the database
    def load_team_season_stats(self, df):
        if df.empty:
            logger.warning("No team-season data to load")
            return 0

        engine = self.get_engine()

        try:
            df.to_sql('team_season_stats', engine, if_exists='replace', index=False)
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM team_season_stats"))
                count = result.scalar()

            logger.info(f"Loaded {count} team-season records")
            return count

        except Exception as e:
            logger.error(f"Error loading team-season stats: {e}")
            return 0

    # Inserts team-game stats into the database
    def load_team_game_stats(self, df):
        if df.empty:
            logger.warning("No team-game data to load")
            return 0

        engine = self.get_engine()

        try:
            df.to_sql('team_game_stats', engine, if_exists='replace', index=False)
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM team_game_stats"))
                count = result.scalar()

            logger.info(f"Loaded {count} team-game records")
            return count

        except Exception as e:
            logger.error(f"Error loading team-game stats: {e}")
            return 0

    # Queries the top N teams for a given season
    def get_top_teams(self, season, limit=10):
        engine = self.get_engine()

        query = text("""
            SELECT season, team, wins, losses, win_percentage, total_points, total_points_allowed
            FROM team_season_stats
            WHERE season = :season
            ORDER BY win_percentage DESC, total_points DESC
            LIMIT :limit
        """)

        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"season": season, "limit": limit})
        return df

    # Queries all stats for a specific team and season
    def get_team_stats(self, team, season):
        engine = self.get_engine()

        query = text("""
            SELECT *
            FROM team_season_stats
            WHERE team = :team AND season = :season
        """)

        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"team": team, "season": season})
        return df


def main():
    # Test with SQLite
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