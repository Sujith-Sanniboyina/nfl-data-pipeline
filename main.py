import logging
from extract.extract import NFLDataExtractor
from transform.transform import TeamStatsAggregator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Runs the full pipeline: extracts play-by-play data, then transforms it into team and season stats
def run_pipeline(seasons):
    logger.info(f"Starting NFL ETL Pipeline for seasons: {seasons}")

    # Step 1: Extract
    logger.info("Step 1: Extract")
    extractor = NFLDataExtractor()
    pbp_data = extractor.fetch_and_save_seasons(seasons)

    if pbp_data.empty:
        logger.error("No data extracted. Pipeline stopping.")
        return None, None

    logger.info(f"Extracted {len(pbp_data)} plays across {len(seasons)} seasons")

    # Step 2: Transform
    logger.info("Step 2: Transform")
    aggregator = TeamStatsAggregator()

    team_games = aggregator.aggregate_game_stats(pbp_data)
    logger.info(f"Created {len(team_games)} team-game records")

    if team_games.empty:
        logger.error("No team-game records created. Pipeline stopping.")
        return pbp_data, None

    season_stats = aggregator.aggregate_season_stats(team_games)
    logger.info(f"Created {len(season_stats)} team-season records")

    # Show a quick preview of results
    print("\nSample Output - Top 10 Teams by Points Scored (2023)")

    if 2023 in season_stats['season'].values:
        top_teams = season_stats[season_stats['season'] == 2023].nlargest(10, 'total_points')
        print(top_teams[['team', 'total_points', 'avg_points', 'wins', 'losses']].to_string(index=False))
    else:
        print("2023 data not available. Showing first 10 rows:")
        print(season_stats.head(10))

    logger.info("Pipeline complete!")

    return team_games, season_stats


def main():
    seasons = [2022, 2023]

    try:
        team_games, season_stats = run_pipeline(seasons)

        print(f"\nPipeline Summary")
        print(f"Extracted:   {len(seasons)} season(s)")
        print(f"Transformed: {len(team_games)} team-game records")
        print(f"Aggregated:  {len(season_stats)} team-season records")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()