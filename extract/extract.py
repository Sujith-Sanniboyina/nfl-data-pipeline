import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Fetches NFL play-by-play data for given seasons and saves them locally
class NFLDataExtractor:

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # Pulls play-by-play data for a list of seasons via nflreadpy
    def fetch_play_by_play(self, seasons):
        try:
            import nflreadpy as nfl

            logger.info(f"Fetching play-by-play data for seasons: {seasons}")
            pbp_df = nfl.load_pbp(seasons).to_pandas()
            logger.info(f"Successfully fetched {len(pbp_df)} plays")

            return pbp_df

        except ImportError:
            logger.error("nflreadpy not installed. Run: pip install 'nflreadpy[pandas]'")
            raise
        except Exception as e:
            logger.error(f"Error fetching NFL data: {e}")
            raise

    # Pulls weekly player-level stats (rushing/receiving/passing) for a list of seasons.
    # This is much lighter than play-by-play and is what the player prop model trains on.
    def fetch_weekly_player_data(self, seasons):
        try:
            import nflreadpy as nfl

            logger.info(f"Fetching weekly player data for seasons: {seasons}")
            weekly_df = nfl.load_player_stats(seasons).to_pandas()
            logger.info(f"Successfully fetched {len(weekly_df)} player-week rows")

            return weekly_df

        except ImportError:
            logger.error("nflreadpy not installed. Run: pip install 'nflreadpy[pandas]'")
            raise
        except Exception as e:
            logger.error(f"Error fetching weekly player data: {e}")
            raise

    # Saves a single season's data to a CSV file
    def save_raw_data(self, df, season, prefix="pbp"):
        filename = self.data_dir / f"{prefix}_{season}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Saved raw data to {filename}")

    # Loops through each season, fetches it, saves it, and returns everything combined
    def fetch_and_save_seasons(self, seasons):
        all_season_data = []

        for season in seasons:
            try:
                logger.info(f"Processing season {season}...")
                season_df = self.fetch_play_by_play([season])
                self.save_raw_data(season_df, season, prefix="pbp")
                all_season_data.append(season_df)

            except Exception as e:
                logger.error(f"Failed to fetch season {season}: {e}")
                continue

        if len(all_season_data) == 0:
            logger.error("No data fetched for any season")
            return pd.DataFrame()

        combined_df = pd.concat(all_season_data, ignore_index=True)
        logger.info(f"Total plays across all seasons: {len(combined_df)}")

        return combined_df

    # Same idea, but for weekly player-level stats instead of play-by-play.
    # A single import_weekly_data(seasons) call covers all seasons at once, so this
    # just slices the result back apart per-season for the raw CSV snapshots.
    def fetch_and_save_weekly_seasons(self, seasons):
        try:
            logger.info(f"Processing weekly player data for seasons {seasons}...")
            weekly_df = self.fetch_weekly_player_data(seasons)

            if weekly_df.empty:
                logger.warning("No weekly player data fetched")
                return pd.DataFrame()

            for season in seasons:
                season_slice = weekly_df[weekly_df["season"] == season]
                if not season_slice.empty:
                    self.save_raw_data(season_slice, season, prefix="weekly")

            logger.info(f"Total player-week rows across all seasons: {len(weekly_df)}")
            return weekly_df

        except Exception as e:
            logger.error(f"Failed to fetch weekly player data: {e}")
            return pd.DataFrame()


def main():
    extractor = NFLDataExtractor()
    # Last 3 completed seasons -- enough history for meaningful rolling features.
    # Once the 2026 season kicks off, add 2026 here too so weekly refreshes pick up
    # the current season's games as they're played.
    seasons = [2023, 2024, 2025]

    try:
        pbp_data = extractor.fetch_and_save_seasons(seasons)
        print(f"Extraction complete! Loaded {len(pbp_data)} plays")
        print(f"Columns available: {list(pbp_data.columns[:10])}...")

        weekly_data = extractor.fetch_and_save_weekly_seasons(seasons)
        print(f"Weekly player data complete! Loaded {len(weekly_data)} player-week rows")

    except Exception as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()