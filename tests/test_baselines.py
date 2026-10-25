import numpy as np
import pytest

from spectralrec.baselines import BiasBaseline, GlobalMean, UserMean


@pytest.fixture
def small_ratings():
    users = np.array([0, 0, 1, 1, 2, 2])
    movies = np.array([0, 1, 0, 2, 0, 1])
    ratings = np.array([2, 5, 1, 4.5, 5, 5])
    return users, movies, ratings


@pytest.fixture
def bias_ratings():
    users = np.array([0, 0, 1, 1])
    movies = np.array([0, 1, 0, 1])
    ratings = np.array([5, 4, 3, 2])
    return users, movies, ratings


def test_global_mean(small_ratings):
    users = np.array([0, 1, 2])
    movies = np.array([2, 1, 2])
    model = GlobalMean(3, 3)
    model.fit(*small_ratings)

    np.testing.assert_allclose(
        model.predict(users, movies), np.array([3.75, 3.75, 3.75])
    )


def test_user_mean(small_ratings):
    users = np.array([0, 1, 2])
    movies = np.array([2, 1, 2])
    model = UserMean(3, 3)
    model.fit(*small_ratings)

    np.testing.assert_allclose(model.predict(users, movies), np.array([3.5, 2.75, 5]))


def test_user_mean_without_ratings(small_ratings):
    users = np.array([0, 1, 2, 3])
    movies = np.array([2, 1, 2, 2])
    model = UserMean(4, 3)
    model.fit(*small_ratings)

    np.testing.assert_allclose(
        model.predict(users, movies), np.array([3.5, 2.75, 5, 3.75])
    )


def test_bias_baseline(bias_ratings):

    model = BiasBaseline(2, 2, reg_movie=0, reg_user=0)
    model.fit(*bias_ratings)
    np.testing.assert_allclose(model.user_bias, [1, -1])
    np.testing.assert_allclose(model.movie_bias, [0.5, -0.5])
    np.testing.assert_allclose(model.predict([0, 0, 1, 1], [0, 1, 0, 1]), [5, 4, 3, 2])


def test_bias_baseline_huge_regularization_terms(bias_ratings):
    model = BiasBaseline(2, 2, reg_movie=1e9, reg_user=1e9)
    model.fit(*bias_ratings)
    np.testing.assert_allclose(
        model.predict([0, 0, 1, 1], [0, 1, 0, 1]), [3.5, 3.5, 3.5, 3.5]
    )


def test_bias_baseline_prediction_size(bias_ratings):
    model = BiasBaseline(2, 3, reg_movie=1e9, reg_user=1e9)
    model.fit(*bias_ratings)
    res = model.predict([0, 0, 0, 1, 1], [0, 1, 0, 0, 1])
    assert res.shape == (5,)


def test_bias_baseline_bias_for_non_existing_movie(bias_ratings):
    model = BiasBaseline(2, 3, reg_movie=2, reg_user=0)
    model.fit(*bias_ratings)
    assert model.movie_bias[2] == 0
