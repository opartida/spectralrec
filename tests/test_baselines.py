import numpy as np
import pytest

from spectralrec.baselines import GlobalMean, UserMean


@pytest.fixture
def small_ratings():
    users = np.array([0, 0, 1, 1, 2, 2])
    movies = np.array([0, 1, 0, 2, 0, 1])
    ratings = np.array([2, 5, 1, 4.5, 5, 5])
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
