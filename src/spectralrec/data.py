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

def check_missing_columns(df: pd.DataFrame, required_columns: set[str], path: Path) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"File {path} is missing columns: {missing}")
