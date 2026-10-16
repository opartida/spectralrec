from pathlib import Path
import pandas as pd


def load_ratings(path: Path) -> pd.DataFrame:
    """Load ratings data from a CSV file."""
    ratings = pd.read_csv(path)
    expected = {
        "userId",
        "movieId",
        "rating",
        "timestamp"
    }
    if(set(ratings.columns) != expected):
        raise ValueError(f"Expected columns {expected}, but got {set(ratings.columns)}")
    return ratings