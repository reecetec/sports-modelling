from sports_modelling.data_management.DataManager import DataManager
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Optional, Union, Dict
import logging
import cbbpy.mens_scraper as s


class DataValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


class NCAABManager(DataManager):
    """Manager for NCAA Basketball data with validation."""

    # Expected columns for each DataFrame
    EXPECTED_COLUMNS = {
        "game_info": [
            "game_id",
            "date",
            "neutral_site",
            "conference_game",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ],
        "box_scores": [
            "game_id",
            "team",
            "player",
            "position",
            "minutes",
            "field_goals_made",
            "field_goals_attempted",
            "three_points_made",
            "three_points_attempted",
            "free_throws_made",
            "free_throws_attempted",
        ],
        "play_by_play": [
            "game_id",
            "half",
            "time_remaining",
            "away_score",
            "home_score",
            "event_type",
            "description",
        ],
    }

    def __init__(self):
        """Initialize NCAABManager with NCAAB-specific data directory."""
        super().__init__(sport_subdir="ncaab")
        self.cbb = s

    def _validate_game_info(self, df: pd.DataFrame) -> None:
        """Validate game info DataFrame."""
        if df.empty:
            raise DataValidationError("Game info DataFrame is empty")

        # Check required columns
        missing_cols = set(self.EXPECTED_COLUMNS["game_info"]) - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns in game_info: {missing_cols}"
            )

        # Validate game_id is unique
        if df["game_id"].duplicated().any():
            raise DataValidationError("Duplicate game_ids found in game_info")

        # Validate scores are non-negative
        if (df["home_score"] < 0).any() or (df["away_score"] < 0).any():
            raise DataValidationError("Negative scores found in game_info")

        # Validate dates
        try:
            pd.to_datetime(df["date"])
        except Exception:
            raise DataValidationError("Invalid dates found in game_info")

    def load_season_data(self, season):
        return {
            "game_info": self.load_parquet(f"game_info/season={season}/data.parquet"),
            "box_scores": self.load_parquet(f"box_scores/season={season}/data.parquet"),
            "play_by_play": self.load_parquet(
                f"play_by_play/season={season}/data.parquet"
            ),
        }

    def fetch_and_save_season(self, season: int) -> Dict[str, pd.DataFrame]:
        """
        Fetch and save all game data for a specific season with validation.
        Data will be saved in /data/ncaab/...

        Args:
            season (int): The season year (e.g., 2023 for 2022-23 season)

        Returns:
            Dict containing the three validated DataFrames
        """
        try:
            self.logger.info(f"Fetching {season-1}-{season} season data...")

            # Get all three DataFrames from get_games_season
            game_info, box_scores, play_by_play = self.cbb.get_games_season(season)

            # Validate each DataFrame
            self.logger.info("Validating data...")
            # self._validate_game_info(game_info)
            # self._validate_box_scores(box_scores)
            # self._validate_play_by_play(play_by_play)

            # Save each DataFrame in ncaab subdirectory
            self.save_parquet(game_info, f"game_info/season={season}/data.parquet")

            self.save_parquet(box_scores, f"box_scores/season={season}/data.parquet")

            self.save_parquet(
                play_by_play, f"play_by_play/season={season}/data.parquet"
            )

            self.logger.info(
                f"Successfully saved all validated data for {season-1}-{season} season"
            )

            return {
                "game_info": game_info,
                "box_scores": box_scores,
                "play_by_play": play_by_play,
            }

        except DataValidationError as e:
            self.logger.error(
                f"Data validation error for {season-1}-{season} season: {str(e)}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Error fetching {season-1}-{season} season data: {str(e)}"
            )
            raise


if __name__ == "__main__":
    ncaab = NCAABManager()
    # ncaab.fetch_and_save_season(2024)
    data = ncaab.load_season_data(2024)
    for df in data.values():
        print('')
        print(df.head())
