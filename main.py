import logging
from extract.extract import NFLDataExtractor
from transform.transform import TeamStatsAggregator
from load.load import NFLDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Runs the full ETL pipeline: extracts play-by-play data, transforms it, then loads it into the database
def run_pipeline(seasons):
    logger.info(f"Starting NFL ETL Pipeline for seasons: {seasons}")

    # Step 1: Extract
    logger.info("Step 1: Extract")
    extractor = NFLDataExtractor()
    pbp_data = extractor.fetch_and_save_seasons(seasons)

    if pbp_data.empty:
        logger.error("No data extracted. Pipeline stopping.")
        return None, None, None, None

    logger.info(f"Extracted {len(pbp_data)} plays across {len(seasons)} seasons")

    # Step 2: Transform
    logger.info("Step 2: Transform")
    aggregator = TeamStatsAggregator()

    team_games = aggregator.aggregate_game_stats(pbp_data)
    logger.info(f"Created {len(team_games)} team-game records")

    season_stats = aggregator.aggregate_season_stats(team_games)
    logger.info(f"Created {len(season_stats)} team-season records")

    # Step 3: Load
    logger.info("Step 3: Load")
    loader = NFLDataLoader()

    try:
        loader.connect()
        loader.create_schema()

        team_games_count = loader.load_team_game_stats(team_games)
        season_stats_count = loader.load_team_season_stats(season_stats)

        logger.info(f"Loaded {team_games_count} team-game records")
        logger.info(f"Loaded {season_stats_count} team-season records")

        top_teams = loader.get_top_teams(2023, limit=10)
        print("\nTop 10 Teams by Win Percentage (2023)")
        print(top_teams.to_string(index=False))

    finally:
        loader.disconnect()

    logger.info("Pipeline complete!")

    return pbp_data, team_games, season_stats, loader


def main():
    seasons = [2022, 2023]

    try:
        pbp_data, team_games, season_stats, loader = run_pipeline(seasons)

        print(f"\nPipeline Summary")
        print(f"Extracted:   {len(pbp_data)} plays")
        print(f"Transformed: {len(team_games)} team-game records")
        print(f"Transformed: {len(season_stats)} team-season records")
        print(f"Loaded to:   data/nfl_data.db")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()