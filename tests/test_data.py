from spectralrec.data import load_ratings, load_movies
import pytest

def test_load_ratings(tmp_path):
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text("userId,movieId,rating,timestamp\n1,1,4.0,964982703\n1,4,4.0,964981247")
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
    movies_path.write_text("movieId,title,genres\n1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n2,Jumanji (1995),Adventure|Children|Fantasy")
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
