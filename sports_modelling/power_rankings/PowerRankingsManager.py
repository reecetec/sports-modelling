from pathlib import Path
import pandas as pd
import numpy as np
from typing import Protocol, Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


class RatingCalculator(Protocol):
    """Protocol defining interface for rating calculators."""

    def calculate_ratings_batch(
        self,
        games_by_date: Dict[pd.Timestamp, pd.DataFrame],
        box_scores_by_game: Dict[str, pd.DataFrame],
        previous_ratings: Optional[Dict[str, float]] = None,
    ) -> Dict[pd.Timestamp, pd.Series]:
        """
        Calculate ratings for multiple dates efficiently.

        Args:
            games_by_date: Dict mapping dates to game DataFrames
            box_scores_by_game: Dict mapping game_ids to box score DataFrames
            previous_ratings: Optional previous ratings to use as starting point
        """
        ...


class BaseRatingCalculator(ABC):
    def _preprocess_data(
        self, game_info: pd.DataFrame, box_scores: pd.DataFrame
    ) -> Tuple[Dict, Dict]:
        """Preprocess data into efficient lookup structures."""
        # Convert game_day to datetime if it isn't already
        game_info = game_info.copy()
        game_info["game_day"] = pd.to_datetime(game_info["game_day"])

        # Group games by date
        games_by_date = {
            date.strftime("%B %d, %Y"): group.copy()
            for date, group in game_info.groupby("game_day")
        }

        # Create box scores lookup
        box_scores_by_game = {
            game_id: group.copy() for game_id, group in box_scores.groupby("game_id")
        }

        return games_by_date, box_scores_by_game


class EloRatingCalculator(BaseRatingCalculator):
    """Calculate Elo ratings for teams."""

    def __init__(
        self,
        k_factor: float = 32,
        home_advantage: float = 100,
        initial_rating: float = 1500,
        margin_multiplier: float = 0.1,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.margin_multiplier = margin_multiplier

    def calculate_ratings_batch(
        self,
        games_by_date: Dict[pd.Timestamp, pd.DataFrame],
        box_scores_by_game: Dict[str, pd.DataFrame],
        previous_ratings: Optional[Dict[str, float]] = None,
    ) -> Dict[pd.Timestamp, pd.Series]:
        ratings = previous_ratings or {}
        ratings_by_date = {}

        # Process dates in chronological order
        for date in sorted(games_by_date.keys()):
            games = games_by_date[date]

            # Process all games for this date
            for _, game in games.iterrows():
                home_team = game["home_team"]
                away_team = game["away_team"]

                # Get or set default ratings
                home_rating = ratings.get(home_team, self.initial_rating)
                away_rating = ratings.get(away_team, self.initial_rating)

                # Calculate expected win probability
                expected = 1 / (
                    1
                    + 10 ** ((away_rating - (home_rating + self.home_advantage)) / 400)
                )

                # Calculate actual outcome and margin factor
                actual = 1 if game["home_score"] > game["away_score"] else 0
                margin = abs(game["home_score"] - game["away_score"])
                margin_factor = np.log(margin + 1) * self.margin_multiplier

                # Adjust K-factor for game importance
                k = self.k_factor
                if game["is_conference"]:
                    k *= 1.1
                if game["is_postseason"]:
                    k *= 1.25

                # Update ratings
                rating_change = k * margin_factor * (actual - expected)
                ratings[home_team] = home_rating + rating_change
                ratings[away_team] = away_rating - rating_change

            # Store ratings for this date
            ratings_by_date[date] = pd.Series(ratings.copy())

        return ratings_by_date


class EfficiencyCalculator(BaseRatingCalculator):
    """Calculate offensive and defensive efficiency ratings."""

    def calculate_ratings_batch(
        self,
        games_by_date: Dict[pd.Timestamp, pd.DataFrame],
        box_scores_by_game: Dict[str, pd.DataFrame],
        previous_ratings: Optional[Dict[str, float]] = None,
    ) -> Dict[pd.Timestamp, pd.Series]:
        # Initialize accumulators for running totals
        team_stats = {}
        efficiency_by_date = {}

        for date in sorted(games_by_date.keys()):
            games = games_by_date[date]

            # Update team stats with games from this date
            for _, game in games.iterrows():
                game_box = box_scores_by_game[game["game_id"]]

                for team, opp_team, score, opp_score in [
                    (
                        game["home_team"],
                        game["away_team"],
                        game["home_score"],
                        game["away_score"],
                    ),
                    (
                        game["away_team"],
                        game["home_team"],
                        game["away_score"],
                        game["home_score"],
                    ),
                ]:
                    if team not in team_stats:
                        team_stats[team] = {
                            "points": 0,
                            "points_allowed": 0,
                            "possessions": 0,
                            "games": 0,
                        }

                    team_box = game_box[game_box["team"] == team]

                    # Calculate possessions
                    possessions = (
                        team_box["fga"].sum()
                        + 0.44 * team_box["fta"].sum()
                        + team_box["to"].sum()
                        - team_box["oreb"].sum()
                    )

                    # Update running totals
                    team_stats[team]["points"] += score
                    team_stats[team]["points_allowed"] += opp_score
                    team_stats[team]["possessions"] += possessions
                    team_stats[team]["games"] += 1

            # Calculate efficiencies for all teams up to this date
            off_ratings = {}
            def_ratings = {}
            for team, stats in team_stats.items():
                if stats["possessions"] > 0:
                    off_ratings[team] = (stats["points"] / stats["possessions"]) * 100
                    def_ratings[team] = (
                        stats["points_allowed"] / stats["possessions"]
                    ) * 100
                else:
                    off_ratings[team] = 0.0
                    def_ratings[team] = 0.0

            efficiency_by_date[date] = {
                "offensive": pd.Series(off_ratings),
                "defensive": pd.Series(def_ratings),
            }

        return efficiency_by_date


class FormRatingCalculator(BaseRatingCalculator):
    """Calculate ratings based on team's recent form."""

    def __init__(self, window_size: int = 5, decay_factor: float = 0.8):
        self.window_size = window_size
        self.decay_factor = decay_factor

    def calculate_ratings_batch(
        self,
        games_by_date: Dict[pd.Timestamp, pd.DataFrame],
        box_scores_by_game: Dict[str, pd.DataFrame],
        previous_ratings: Optional[Dict[str, float]] = None,
    ) -> Dict[pd.Timestamp, pd.Series]:
        # Initialize storage for team game histories and ratings by date
        team_games: Dict[str, List] = {}
        form_by_date = {}

        # Process dates chronologically
        for date in sorted(games_by_date.keys()):
            games = games_by_date[date]

            # Process each game for this date
            for _, game in games.iterrows():
                game_box = box_scores_by_game[game["game_id"]]

                # Calculate game metrics for both teams
                for team, opp_team, score, opp_score in [
                    (
                        game["home_team"],
                        game["away_team"],
                        game["home_score"],
                        game["away_score"],
                    ),
                    (
                        game["away_team"],
                        game["home_team"],
                        game["away_score"],
                        game["home_score"],
                    ),
                ]:
                    team_box = game_box[game_box["team"] == team]
                    opp_box = game_box[game_box["team"] == opp_team]

                    # Calculate performance metrics
                    margin = score - opp_score
                    win = 1 if margin > 0 else 0

                    # Shooting efficiency
                    fg_attempts = team_box["fga"].sum()
                    shooting_pct = (
                        (team_box["fgm"].sum() / fg_attempts * 100)
                        if fg_attempts > 0
                        else 0
                    )

                    # Rebounding
                    team_rebounds = team_box["reb"].sum()
                    opp_rebounds = opp_box["reb"].sum()
                    rebound_margin = team_rebounds - opp_rebounds

                    # Calculate game score
                    game_score = (
                        win * 50  # Base points for win
                        + margin * 2  # Points for margin
                        + (shooting_pct - 50) * 0.5  # Points for shooting above 50%
                        + rebound_margin * 0.5  # Points for rebound margin
                        + previous_ratings.get(opp_team, 0) * 0.3
                        if previous_ratings
                        else 0  # Opponent strength
                    )

                    # Adjust for game importance
                    if game["is_conference"]:
                        game_score *= 1.1
                    if game["is_postseason"]:
                        game_score *= 1.25

                    # Add to team's game history
                    if team not in team_games:
                        team_games[team] = []
                    team_games[team].append((date, game_score))

                    # Keep only recent games
                    if len(team_games[team]) > self.window_size:
                        team_games[team] = team_games[team][-self.window_size :]

            # Calculate form ratings for all teams up to this date
            form_ratings = {}
            for team, games in team_games.items():
                if not games:
                    form_ratings[team] = 0.0
                    continue

                # Calculate weighted average of recent game scores
                scores = [score for _, score in games]
                weights = [self.decay_factor**i for i in range(len(scores) - 1, -1, -1)]
                form_ratings[team] = np.average(scores, weights=weights)

            form_by_date[date] = pd.Series(form_ratings)

        return form_by_date


class PowerRankingsManager:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.rankings_dir = root_dir / "rankings"
        self.rankings_dir.mkdir(parents=True, exist_ok=True)

        # Initialize calculators
        self.calculators = {
            "elo": EloRatingCalculator(),
            "efficiency": EfficiencyCalculator(),
            "form": FormRatingCalculator(),
        }

        self.scaler = StandardScaler()
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_calculator(self, name: str, calculator: RatingCalculator):
        self.calculators[name] = calculator

    def calculate_historical_rankings(
        self, game_info: pd.DataFrame, box_scores: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate rankings efficiently using batch processing."""
        self.logger.info("Preprocessing data...")

        # Preprocess data into efficient lookup structures
        games_by_date, box_scores_by_game = self.calculators["elo"]._preprocess_data(
            game_info, box_scores
        )

        # Calculate all ratings
        self.logger.info("Calculating ratings...")
        all_ratings = {}

        for name, calculator in self.calculators.items():
            self.logger.info(f"Calculating {name} ratings...")
            ratings_by_date = calculator.calculate_ratings_batch(
                games_by_date, box_scores_by_game
            )

            if isinstance(ratings_by_date[next(iter(ratings_by_date))], dict):
                # Handle multiple rating types from one calculator
                for rating_type, ratings in ratings_by_date[
                    next(iter(ratings_by_date))
                ].items():
                    all_ratings[f"{name}_{rating_type}"] = ratings_by_date
            else:
                all_ratings[name] = ratings_by_date

        # Combine all ratings into DataFrame
        self.logger.info("Combining ratings...")
        rankings_data = []

        for date in sorted(games_by_date.keys()):
            daily_ratings = {}
            for rating_name, ratings_dict in all_ratings.items():
                if isinstance(ratings_dict[date], dict):
                    for subtype, ratings in ratings_dict[date].items():
                        daily_ratings[f"{rating_name}_{subtype}"] = ratings
                else:
                    daily_ratings[rating_name] = ratings_dict[date]

            daily_df = pd.DataFrame(daily_ratings)
            daily_df["date"] = date  # Date is already in correct format
            daily_df = daily_df.reset_index().rename(columns={"index": "team"})

            # Add standardized versions
            rating_columns = [
                col for col in daily_df.columns if col not in ["team", "date"]
            ]
            if rating_columns:
                scaled_ratings = self.scaler.fit_transform(daily_df[rating_columns])
                for col, scaled_col in zip(rating_columns, scaled_ratings.T):
                    daily_df[f"{col}_standardized"] = scaled_col

            rankings_data.append(daily_df)

        # Combine all dates
        self.logger.info("Finalizing rankings...")
        return pd.concat(rankings_data, ignore_index=True)

    def save_rankings(self, rankings_df: pd.DataFrame, season: int):
        filepath = self.rankings_dir / f"rankings_{season}.parquet"
        rankings_df.to_parquet(filepath)
        self.logger.info(f"Saved rankings for season {season}")

    def load_rankings(self, season: int) -> pd.DataFrame:
        filepath = self.rankings_dir / f"rankings_{season}.parquet"
        try:
            return pd.read_parquet(filepath)
        except FileNotFoundError:
            self.logger.warning(f"No rankings file found for season {season}")
            return pd.DataFrame()

    def get_rankings_for_date(self, date: pd.Timestamp, season: int) -> pd.DataFrame:
        """
        Get rankings for a specific date.

        Args:
            date: Date to get rankings for (pd.Timestamp)
            season: Season year

        Returns:
            DataFrame with rankings for specified date
        """
        rankings_df = self.load_rankings(season)
        if rankings_df.empty:
            return pd.DataFrame()

        # Convert input timestamp to the same format as stored dates
        date_str = date.strftime("%B %d, %Y")

        return rankings_df[rankings_df["date"] == date_str].copy()

    def get_team_ranking_history(self, team: str, season: int) -> pd.DataFrame:
        """
        Get historical rankings for a specific team.

        Args:
            team: Team name
            season: Season year

        Returns:
            DataFrame with team's rankings over time
        """
        rankings_df = self.load_rankings(season)
        if rankings_df.empty:
            return pd.DataFrame()

        return rankings_df[rankings_df["team"] == team].sort_values("date")

    def get_rankings_date_range(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp, season: int
    ) -> pd.DataFrame:
        """
        Get rankings for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            season: Season year

        Returns:
            DataFrame with rankings between start_date and end_date
        """
        rankings_df = self.load_rankings(season)
        if rankings_df.empty:
            return pd.DataFrame()

        # Convert dates to string format
        start_str = start_date.strftime("%B %d, %Y")
        end_str = end_date.strftime("%B %d, %Y")

        mask = (rankings_df["date"] >= start_str) & (rankings_df["date"] <= end_str)
        return rankings_df[mask].copy()
