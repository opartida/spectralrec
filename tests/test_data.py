import pandas as pd
import pytest

from spectralrec.data import load_movies, load_ratings, validate_ratings


def test_load_ratings(tmp_path):
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "userId,movieId,rating,timestamp\n1,1,4.0,964982703\n1,4,4.0,964981247"
    )
    ratings = load_ratings(ratings_path)
    assert ratings.shape[0] == 2
    assert set(ratings.columns) == {"userId", "movieId", "rating", "timestamp"}


def test_load_ratings_missing_columns(tmp_path):
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("userId,movieId,rating\n1,1,4.0\n1,3,4.0")
    with pytest.raises(ValueError) as excinfo:
        load_ratings(ratings_path)
    assert str(ratings_path) in str(excinfo.value)
    assert "timestamp" in str(excinfo.value)


def test_load_movies(tmp_path):
    movies_path = tmp_path / "movies.csv"
    movies_path.write_text(
        "movieId,title,genres\n1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n2,Jumanji (1995),Adventure|Children|Fantasy"
    )
    movies = load_movies(movies_path)
    assert movies.shape[0] == 2
    assert set(movies.columns) == {"movieId", "title", "genres"}


def test_load_movies_missing_columns(tmp_path):
    movies_path = tmp_path / "movies.csv"
    movies_path.write_text("movieId,title\n1,Toy Story (1995)\n2,Jumanji (1995)")
    with pytest.raises(ValueError) as excinfo:
        load_movies(movies_path)
    assert str(movies_path) in str(excinfo.value)
    assert "genres" in str(excinfo.value)


def test_validate_ratings():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [1, 2, 3],
            "rating": [4.0, 5.0, 3.5],
            "timestamp": [964982703, 964981247, 964982224],
        }
    )
    # This should not raise any exceptions
    validate_ratings(ratings)


@pytest.mark.parametrize("column", ["userId", "movieId", "rating", "timestamp"])
def test_validate_ratings_with_null_values(column):
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [1, 2, 3],
            "rating": [4.0, 5.0, 3.5],
            "timestamp": [964982703, 964981247, 964982224],
        }
    )
    ratings[column] = ratings[column].astype(float)
    ratings.loc[1, column] = None
    with pytest.raises(ValueError) as excinfo:
        validate_ratings(ratings)
    assert column in str(excinfo.value)
    assert "null values" in str(excinfo.value)


def test_validate_ratings_accepts_ratings_in_range():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [1, 2, 3],
            "rating": [0.5, 4.0, 5.0],  # Valid rating values
            "timestamp": [964982703, 964981247, 964982224],
        }
    )
    # This should not raise any exceptions
    validate_ratings(ratings)


def test_validate_ratings_rejects_ratings_out_of_range():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [1, 2, 3],
            "rating": [4.0, 6.0, 3.5],  # Invalid rating value (6.0)
            "timestamp": [964982703, 964981247, 964982224],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        validate_ratings(ratings)
    assert "Ratings must be between 0.5 and 5.0" in str(excinfo.value)


def test_validate_ratings_half_increments():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [1, 2, 3],
            "rating": [4.0, 4.5, 3.5],  # Valid half-increment rating values
            "timestamp": [964982703, 964981247, 964982224],
        }
    )
    # This should not raise any exceptions
    validate_ratings(ratings)


def test_validate_ratings_rejects_non_half_increments():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [1, 2, 3],
            "rating": [4.0, 4.3, 3.5],  # Invalid rating value (4.3)
            "timestamp": [964982703, 964981247, 964982224],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        validate_ratings(ratings)
    assert "Ratings must be in increments of 0.5" in str(excinfo.value)


def test_validate_ratings_rejects_non_numeric_ratings():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [1, 2, 3],
            "rating": [4.0, "five", 3.5],  # Invalid rating value ("five")
            "timestamp": [964982703, 964981247, 964982224],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        validate_ratings(ratings)
    assert "Ratings must be numeric" in str(excinfo.value)


def test_validate_ratings_rejects_duplicate_movieId_userId_pairs():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [1, 1, 2],
            "rating": [4.0, 5.0, 3.5],  # Duplicate (userId=1, movieId=1)
            "timestamp": [964982703, 964981247, 964982224],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        validate_ratings(ratings)
    assert "Duplicate ratings found for the same user and movie combination" in str(
        excinfo.value
    )


def test_validate_ratings_rejects_non_integer_userId_movieId():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2.5, 3],  # Invalid userId (2.5)
            "movieId": [1, 2, 3],
            "rating": [4.0, 5.0, 3.5],
            "timestamp": [964982703, 964981247, 964982224],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        validate_ratings(ratings)
    assert "All userId values must be integers" in str(excinfo.value)

    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [1, 2.5, 3],  # Invalid movieId (2.5)
            "rating": [4.0, 5.0, 3.5],
            "timestamp": [964982703, 964981247, 964982224],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        validate_ratings(ratings)
    assert "All movieId values must be integers" in str(excinfo.value)
