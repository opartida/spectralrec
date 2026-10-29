import numpy as np

from spectralrec.lowrank import predict_observed, to_dense


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
