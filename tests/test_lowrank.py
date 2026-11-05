import numpy as np
import pytest

from spectralrec.lowrank import (
    TruncatedSVD,
    compute_singular_values,
    predict_observed,
    to_dense,
)


def test_predict_observed():
    user_factors = np.array([[0.5, 3], [3, 0.2]])
    movie_factors = np.array([[5, 3], [3, 1], [0.2, 0.5]])
    users = np.array([0, 0, 0, 1, 1])
    movies = np.array([0, 1, 2, 1, 2])
    result = predict_observed(
        user_factors=user_factors,
        movie_factors=movie_factors,
        users=users,
        movies=movies,
    )
    np.testing.assert_array_almost_equal(result, [11.5, 4.5, 1.6, 9.2, 0.7])


def test_to_dense():
    users = [1, 2]
    movies = [2, 3]
    values = [3, 6]
    expected_result = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 6, 0],
        ]
    )
    np.testing.assert_array_equal(
        to_dense(users=users, movies=movies, values=values, shape=(3, 5)),
        expected_result,
    )


def test_rank_3_matrix_has_three_singular_values():
    rng = np.random.default_rng(42)
    B = rng.normal(size=(20, 3))
    C = rng.normal(size=(15, 3))

    M = B @ np.transpose(C)

    singular_values = compute_singular_values(M)

    assert np.count_nonzero(singular_values > 1e-10) == 3


def test_sum_of_squares_equals_sum_of_squared_singular_values():
    rng = np.random.default_rng(42)
    B = rng.normal(size=(20, 15))

    singular_values = compute_singular_values(B)

    assert np.sum(B**2) == pytest.approx(np.sum(singular_values**2))


def test_truncated_svd_reconstructs_rank_2_matrix():
    rank = 2
    k = 2
    model = TruncatedSVD(k)
    rng = np.random.default_rng(42)
    shape = (20, 15)
    n_users = shape[0]
    n_movies = shape[1]
    B = rng.normal(size=(n_users, rank))
    C = rng.normal(size=(n_movies, rank))
    M = B @ np.transpose(C)
    users = np.repeat(np.arange(n_users), n_movies)
    movies = np.tile(np.arange(n_movies), n_users)
    values = np.ravel(M)
    model.fit(users=users, movies=movies, residuals=values, shape=shape)
    np.testing.assert_array_almost_equal(
        model.predict(users=users, movies=movies), values
    )


def test_truncated_svd_error_equals_sum_of_discarded_squared_singular_values():
    rank = 2
    k = 1
    model = TruncatedSVD(k)
    rng = np.random.default_rng(42)
    shape = (20, 15)
    n_users = shape[0]
    n_movies = shape[1]
    B = rng.normal(size=(n_users, rank))
    C = rng.normal(size=(n_movies, rank))
    M = B @ np.transpose(C)
    users = np.repeat(np.arange(n_users), n_movies)
    movies = np.tile(np.arange(n_movies), n_users)
    values = np.ravel(M)
    model.fit(users=users, movies=movies, residuals=values, shape=shape)
    P = model.predict(users=users, movies=movies)
    e = np.sum((values - P) ** 2)
    singular_values = compute_singular_values(M)
    np.testing.assert_almost_equal(e, np.sum(singular_values[k:] ** 2))
