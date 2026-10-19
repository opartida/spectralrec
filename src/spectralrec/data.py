from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RATING_COLUMNS = {"userId", "movieId", "rating", "timestamp"}
MOVIE_COLUMNS = {"movieId", "title", "genres"}


def load_ratings(path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(path)
    check_missing_columns(ratings, RATING_COLUMNS, path)
    return ratings


def load_movies(path: Path) -> pd.DataFrame:
    movies = pd.read_csv(path)
    check_missing_columns(movies, MOVIE_COLUMNS, path)
    return movies


def check_missing_columns(
    df: pd.DataFrame, required_columns: set[str], path: Path
) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"File {path} is missing columns: {missing}")


def validate_null_values(df: pd.DataFrame, required_columns: set[str]) -> None:
    for column in required_columns:
        if df[column].isnull().any():
            raise ValueError(f"DataFrame column {column} has null values")


def validate_ratings(ratings: pd.DataFrame) -> None:
    validate_null_values(ratings, RATING_COLUMNS)
    if not pd.api.types.is_numeric_dtype(ratings["rating"]):
        raise ValueError("Ratings must be numeric")
    if not ratings["rating"].between(0.5, 5.0).all():
        raise ValueError("Ratings must be between 0.5 and 5.0")
    if not ratings["rating"].apply(lambda x: x % 0.5 == 0).all():
        raise ValueError("Ratings must be in increments of 0.5")
    if not ratings["userId"].apply(lambda x: isinstance(x, int)).all():
        raise ValueError("All userId values must be integers")
    if not ratings["movieId"].apply(lambda x: isinstance(x, int)).all():
        raise ValueError("All movieId values must be integers")
    if ratings.duplicated(subset=["userId", "movieId"]).any():
        raise ValueError(
            "Duplicate ratings found for the same user and movie combination"
        )


@dataclass
class IndexMaps:
    user_ids: np.ndarray  # position -> userId
    movie_ids: np.ndarray  # position -> movieId
    user_to_index: dict  # userId -> position
    movie_to_index: dict  # movieId -> position


def build_index_maps(
    train: pd.DataFrame,
) -> IndexMaps:
    user_ids = np.sort(train["userId"].unique())
    movie_ids = np.sort(train["movieId"].unique())
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    return IndexMaps(user_ids, movie_ids, user_to_index, movie_to_index)


def encode_ratings(df: pd.DataFrame, maps: IndexMaps):
    """Encode user/movie IDs into positional indexes.

    Returns a tuple of three arrays: users, movies, ratings, each with length N.
    Raises a clear error if a row contains an ID not present in the index maps.
    """

    users = df["userId"].map(maps.user_to_index)
    movies = df["movieId"].map(maps.movie_to_index)
    ratings = df["rating"]

    users_without_index = users.isna()
    movies_without_index = movies.isna()

    if users_without_index.any():
        raise ValueError(
            f"Unknown userId values (not in index maps): {df.loc[users_without_index, 'userId'].unique()}"
        )

    if movies_without_index.any():
        raise ValueError(
            f"Unknown movieId values (not in index maps): {df.loc[movies_without_index, 'movieId'].unique()}"
        )
    return users.to_numpy(), movies.to_numpy(), ratings.to_numpy()


def split_ratings(df, seed: int, ratios=(0.8, 0.1, 0.1)):
    if not np.isclose(sum(ratios), 1):
        raise ValueError("ratios do not add up to one")
    rng = np.random.default_rng(seed)
    nrows = df.shape[0]
    order = rng.permutation(nrows)
    num_rows_in_training = int(ratios[0] * nrows)
    num_rows_in_validation = int(ratios[1] * nrows)
    training = df.iloc[order[0:num_rows_in_training]]
    validation = df.iloc[
        order[num_rows_in_training : num_rows_in_training + num_rows_in_validation]
    ]
    testing = df.iloc[order[num_rows_in_training + num_rows_in_validation :]]

    return training, validation, testing


def drop_unknown(df, train):
    known_users = df["userId"].isin(train["userId"].unique())
    known_movies = df["movieId"].isin(train["movieId"].unique())

    df_filtered = df[known_users & known_movies]

    num_rows_deleted = df.shape[0] - df_filtered.shape[0]

    return df_filtered, num_rows_deleted
