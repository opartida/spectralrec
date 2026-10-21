import numpy as np
import pandas as pd
import pytest

from spectralrec.data import (
    build_index_maps,
    drop_unknown,
    encode_ratings,
    load_movies,
    load_ratings,
    split_ratings,
    validate_ratings,
)


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


def test_build_index_maps_sorted_movies_and_users():
    ratings = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30],
            "movieId": [7, 99, 7, 99],
            "rating": [4.0, 5.0, 3.5, 5],
        }
    )

    maps = build_index_maps(ratings)

    np.testing.assert_array_equal(maps.user_ids, [10, 30])
    assert maps.user_to_index == {10: 0, 30: 1}

    np.testing.assert_array_equal(maps.movie_ids, [7, 99])
    assert maps.movie_to_index == {7: 0, 99: 1}


def test_encode_ratings():
    ratings = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30],
            "movieId": [7, 99, 7, 99],
            "rating": [4.0, 5.0, 3.5, 5],
        }
    )

    maps = build_index_maps(ratings)
    (users, movies, ratings) = encode_ratings(ratings, maps)

    np.testing.assert_array_equal(users, [1, 0, 0, 1])
    np.testing.assert_array_equal(maps.user_ids[users], [30, 10, 10, 30])
    np.testing.assert_array_equal(movies, [0, 1, 0, 1])
    np.testing.assert_array_equal(maps.movie_ids[movies], [7, 99, 7, 99])


def test_encode_ratings_userid_without_index():
    ratings = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30],
            "movieId": [7, 99, 7, 99],
            "rating": [4.0, 5.0, 3.5, 5],
        }
    )
    ratings_test = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30, 27],
            "movieId": [7, 99, 7, 99, 99],
            "rating": [4.0, 5.0, 3.5, 5, 5],
        }
    )
    maps = build_index_maps(ratings)

    with pytest.raises(ValueError) as excinfo:
        encode_ratings(ratings_test, maps)
    assert "Unknown userId values (not in index maps)" in str(excinfo.value)
    assert "27" in str(excinfo.value)


def test_split_ratings():
    ratings = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30, 20, 20, 21, 21, 22, 22],
            "movieId": [7, 99, 7, 99, 7, 99, 7, 99, 7, 99],
            "rating": [4.0, 5.0, 3.5, 5, 4.0, 5.0, 3.5, 5, 5, 1],
        }
    )

    (train, val, test) = split_ratings(ratings, 40, (0.8, 0.1, 0.1))

    assert len(train.index.intersection(val.index)) == 0
    assert len(val.index.intersection(test.index)) == 0
    assert len(train.index.intersection(test.index)) == 0
    assert train.shape[0] + val.shape[0] + test.shape[0] == 10


def test_split_ratings_same_seed():
    ratings = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30, 20, 20, 21, 21, 22, 22],
            "movieId": [7, 99, 7, 99, 7, 99, 7, 99, 7, 99],
            "rating": [4.0, 5.0, 3.5, 5, 4.0, 5.0, 3.5, 5, 5, 1],
        }
    )

    (train1, val1, test1) = split_ratings(ratings, 40, (0.8, 0.1, 0.1))
    (train2, val2, test2) = split_ratings(ratings, 40, (0.8, 0.1, 0.1))

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)


def test_split_ratings_different_seed():
    ratings = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30, 20, 20, 21, 21, 22, 22],
            "movieId": [7, 99, 7, 99, 7, 99, 7, 99, 7, 99],
            "rating": [4.0, 5.0, 3.5, 5, 4.0, 5.0, 3.5, 5, 5, 1],
        }
    )
    (train1, _, _) = split_ratings(ratings, 40, (0.8, 0.1, 0.1))
    (train2, _, _) = split_ratings(ratings, 41, (0.8, 0.1, 0.1))

    assert set(train1.index) != set(train2.index)


def test_split_ratings_ratios_dont_add_up_to_one():
    ratings = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30, 20, 20, 21, 21, 22, 22],
            "movieId": [7, 99, 7, 99, 7, 99, 7, 99, 7, 99],
            "rating": [4.0, 5.0, 3.5, 5, 4.0, 5.0, 3.5, 5, 5, 1],
        }
    )
    with pytest.raises(ValueError) as execinfo:
        split_ratings(ratings, 40, (0.8, 0.1, 0.2))
    assert "ratios do not add up to one" in str(execinfo.value)


def test_drop_unknown():
    train = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30, 20, 20, 21, 21, 22, 22],
            "movieId": [7, 99, 7, 99, 7, 99, 7, 99, 7, 99],
            "rating": [4.0, 5.0, 3.5, 5, 4.0, 5.0, 3.5, 5, 5, 1],
        }
    )

    val = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30, 20, 20, 21, 21, 22, 51],
            "movieId": [7, 99, 7, 99, 7, 99, 7, 99, 100, 99],
            "rating": [4.0, 5.0, 3.5, 5, 4.0, 5.0, 3.5, 5, 5, 1],
        }
    )

    expected_filtered = pd.DataFrame(
        {
            "userId": [30, 10, 10, 30, 20, 20, 21, 21],
            "movieId": [7, 99, 7, 99, 7, 99, 7, 99],
            "rating": [4.0, 5.0, 3.5, 5, 4.0, 5.0, 3.5, 5],
        }
    )

    (df_filtered, num_rows_deleted) = drop_unknown(val, train)

    pd.testing.assert_frame_equal(df_filtered, expected_filtered)
    assert num_rows_deleted == 2
