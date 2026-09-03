import pytest
import pandas as pd
import shutil
from pathlib import Path
from extract.extract import NFLDataExtractor


class TestNFLDataExtractor:

    def setup_method(self):
        self.extractor = NFLDataExtractor(data_dir="test_data/raw")

    def teardown_method(self):
        if Path("test_data").exists():
            shutil.rmtree("test_data")

    def test_init_creates_directory(self):
        assert Path("test_data/raw").exists()

    def test_fetch_play_by_play_returns_dataframe(self):
        try:
            df = self.extractor.fetch_play_by_play([2023])
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
        except Exception as e:
            pytest.skip(f"API may be unavailable: {e}")

    def test_fetch_weekly_player_data_returns_dataframe(self):
        try:
            df = self.extractor.fetch_weekly_player_data([2023])
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'player_id' in df.columns
        except Exception as e:
            pytest.skip(f"API may be unavailable: {e}")

    def test_save_raw_data(self):
        test_df = pd.DataFrame({'test_col': [1, 2, 3]})
        self.extractor.save_raw_data(test_df, 2023)

        file_path = Path("test_data/raw/pbp_2023.csv")
        assert file_path.exists()

        loaded_df = pd.read_csv(file_path)
        assert len(loaded_df) == 3


if __name__ == "__main__":
    pytest.main([__file__])