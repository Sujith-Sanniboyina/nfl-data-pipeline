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

    # Pulls play-by-play data for a list of seasons from nfl_data_py
    def fetch_play_by_play(self, seasons):
        try:
            import nfl_data_py as nfl

            logger.info(f"Fetching play-by-play data for seasons: {seasons}")
            pbp_df = nfl.import_pbp_data(seasons)
            logger.info(f"Successfully fetched {len(pbp_df)} plays")

            return pbp_df

        except ImportError:
            logger.error("nfl_data_py not installed. Run: pip install nfl_data_py")
            raise
        except Exception as e:
            logger.error(f"Error fetching NFL data: {e}")
            raise

    # Saves a single season's data to a CSV file
    def save_raw_data(self, df, season):
        filename = self.data_dir / f"pbp_{season}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Saved raw data to {filename}")

    # Loops through each season, fetches it, saves it, and returns everything combined
    def fetch_and_save_seasons(self, seasons):
        all_season_data = []

        for season in seasons:
            try:
                logger.info(f"Processing season {season}...")
                season_df = self.fetch_play_by_play([season])
                self.save_raw_data(season_df, season)
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


def main():
    extractor = NFLDataExtractor()
    seasons = [2022, 2023]

    try:
        pbp_data = extractor.fetch_and_save_seasons(seasons)
        print(f"Extraction complete! Loaded {len(pbp_data)} plays")
        print(f"Columns available: {list(pbp_data.columns[:10])}...")

    except Exception as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()