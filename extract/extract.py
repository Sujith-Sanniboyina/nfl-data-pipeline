import pandas as pd
from typing import Optional, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This class fetches NFL data for a season saves the data locally
class NFLDataExtractor:
    #initializng the extraction process
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    #Gets play-by-play date for any given season    
    def fetch_play_by_play(self, seasons: list) -> pd.DataFrame:
        try:
            import nfl_data_py as nfl
            logger.info(f"Fetching play-by-play data for seasons: {seasons}")
            
            # Fetch the data
            pbp_df = nfl.import_pbp_data(seasons)
            
            logger.info(f"Successfully fetched {len(pbp_df)} plays")
            return pbp_df
            
        except ImportError:
            logger.error("nfl_data_py not installed. Run: pip install nfl_data_py")
            raise
        except Exception as e:
            logger.error(f"Error fetching NFL data: {e}")
            raise
    #Saves the data into a CSV file for easy use
    def save_raw_data(self, df: pd.DataFrame, season: int) -> None:
        filename = self.data_dir / f"pbp_{season}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Saved raw data to {filename}")
    
    #Gets multiple years of data and saves each one individually
    def fetch_and_save_seasons(self, seasons: list) -> pd.DataFrame:
        all_data = []
        
        for season in seasons:
            try:
                logger.info(f"Processing season {season}...")
                season_df = self.fetch_play_by_play([season])
                self.save_raw_data(season_df, season)
                all_data.append(season_df)
            except Exception as e:
                logger.error(f"Failed to fetch season {season}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total plays across all seasons: {len(combined_df)}")
            return combined_df
        else:
            logger.error("No data fetched for any season")
            return pd.DataFrame()


def main():
    # Test with just one season (2023) for MVP
    extractor = NFLDataExtractor()
    
    # For MVP, use just 2-3 seasons as mentioned in requirements
    seasons = [2023]  # Testing functionality with 2023 season
    seasons = [2022, 2023]  # Uncomment later
    
    try:
        pbp_data = extractor.fetch_and_save_seasons(seasons)
        print(f"Extraction complete! Loaded {len(pbp_data)} plays")
        print(f"Columns available: {list(pbp_data.columns[:10])}...")  # Show first 10 columns
    except Exception as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()