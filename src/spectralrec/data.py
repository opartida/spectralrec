from pathlib import Path

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
