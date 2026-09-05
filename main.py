import logging
from extract.extract import NFLDataExtractor
from transform.transform import TeamStatsAggregator, PlayerStatsTransformer
from load.load import NFLDataLoader
from analytics.analytics import NFLReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Runs the full ETL pipeline: extracts play-by-play + weekly player data, transforms
# both, then loads everything into the database
def run_pipeline(seasons):
    logger.info(f"Starting NFL ETL Pipeline for seasons: {seasons}")

    # Step 1: Extract
    logger.info("Step 1: Extract")
    extractor = NFLDataExtractor()
    pbp_data = extractor.fetch_and_save_seasons(seasons)

    if pbp_data.empty:
        logger.error("No data extracted. Pipeline stopping.")
        return None, None, None, None, None

    logger.info(f"Extracted {len(pbp_data)} plays across {len(seasons)} seasons")

    # Step 1b: Extract weekly player-level data (for the prop predictor model)
    weekly_data = extractor.fetch_and_save_weekly_seasons(seasons)
    logger.info(f"Extracted {len(weekly_data)} player-week rows across {len(seasons)} seasons")

    # Step 1c: Schedules -- needed so PlayerStatsTransformer can correctly set
    # home_game (previously always left NULL/False; see transform.py).
    import nflreadpy as nfl
    schedules = nfl.load_schedules(seasons).to_pandas()
    logger.info(f"Loaded {len(schedules)} scheduled games across {len(seasons)} seasons")

    # Step 2: Transform
    logger.info("Step 2: Transform")
    aggregator = TeamStatsAggregator()

    team_games = aggregator.aggregate_game_stats(pbp_data)
    logger.info(f"Created {len(team_games)} team-game records")

    season_stats = aggregator.aggregate_season_stats(team_games)
    logger.info(f"Created {len(season_stats)} team-season records")

    player_transformer = PlayerStatsTransformer()
    player_game_stats = player_transformer.prepare_player_game_stats(weekly_data, schedules_df=schedules)
    logger.info(f"Created {len(player_game_stats)} player-game records")

    # Step 3: Load
    logger.info("Step 3: Load")
    loader = NFLDataLoader()

    try:
        loader.connect()
        loader.create_schema()

        team_games_count = loader.load_team_game_stats(team_games)
        season_stats_count = loader.load_team_season_stats(season_stats)
        player_stats_count = loader.load_player_game_stats(player_game_stats)

        logger.info(f"Loaded {team_games_count} team-game records")
        logger.info(f"Loaded {season_stats_count} team-season records")
        logger.info(f"Loaded {player_stats_count} player-game records")

        # Step 4: Analytics
        logger.info("Step 4: Analytics")
        reporter = NFLReportGenerator(loader=loader)

        # Generate top teams report for the most recent season in this run
        latest_season = max(seasons)
        top_teams_report = reporter.generate_top_teams_report(latest_season)
        logger.info(f"Generated top teams report")

    finally:
        loader.disconnect()

    logger.info("Pipeline complete!")

    return pbp_data, team_games, season_stats, player_game_stats, loader


def main():
    # 2019 onward: train.py needs several seasons before CALIBRATION_SEASON (2023)
    # to train the base model on, plus 2023 (calibration) and 2024 (final test) --
    # see model/data.py. 2025 is included too so the DB stays current for live
    # predictions once that season's weekly data is available. Previously this only
    # loaded [2023, 2024, 2025], which meant calibrate.py/backtest.py/evaluate.py --
    # which all read from this DB -- only ever saw 2023-2025, a much smaller and
    # different dataset than what train.py was actually training the model on.
    seasons = list(range(2019, 2027))

    try:
        pbp_data, team_games, season_stats, player_game_stats, loader = run_pipeline(seasons)

        print(f"\n" + "=" * 50)
        print("PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Extracted:   {len(pbp_data)} plays")
        print(f"Transformed: {len(team_games)} team-game records")
        print(f"Transformed: {len(season_stats)} team-season records")
        print(f"Transformed: {len(player_game_stats)} player-game records")
        print(f"Loaded to:   Supabase (Postgres)")
        print(f"Reports:     top_teams_report.txt")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()