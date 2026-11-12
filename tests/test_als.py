import numpy as np
import pytest

from spectralrec.als import (
    als_loss,
    group_indices,
    solve_ridge,
    update_movies,
    update_users,
)
from spectralrec.sgd import init_factor_model


def test_solve_ridge_without_reg():
    F = np.array([[1, 0], [0, 1], [1, 1]])
    r = np.array([0.5, -0.2, 0.6])
    reg = 0

    Uu = solve_ridge(F=F, r=r, reg=reg)

    np.testing.assert_almost_equal(Uu, [0.6, -0.1])


def test_solve_ridge_with_reg():
    F = np.array([[1, 0], [0, 1], [1, 1]])
    r = np.array([0.5, -0.2, 0.6])
    reg = 0.5

    Uu_ridge = solve_ridge(F=F, r=r, reg=reg)

    F_aug = np.vstack((F, np.sqrt(reg) * np.eye(N=F.shape[1], M=F.shape[1])))
    r_aug = np.concatenate((r, np.zeros(shape=(F.shape[1],))))

    x = np.linalg.lstsq(F_aug, r_aug, rcond=None)

    np.testing.assert_almost_equal(Uu_ridge, x[0])


def test_group_indices():
    ids = [2, 0, 2, 1, 0]
    n = 3
    expected = [
        np.array([1, 4]),
        np.array([3]),
        np.array([0, 2]),
    ]
    result = group_indices(ids=ids, n=n)

    for group, expected_group in zip(result, expected):
        np.testing.assert_array_equal(group, expected_group)
    assert len(result) == n
    np.testing.assert_array_equal(np.sort(np.concatenate(result)), np.arange(len(ids)))


def test_group_indices_empty_group():
    ids = [2, 0, 2, 1, 0]
    n = 4
    result = group_indices(ids=ids, n=n)
    assert len(result[3]) == 0


@pytest.mark.parametrize("weighted", [True, False])
def test_als_half_steps_do_not_increase_loss(weighted):
    rng = np.random.default_rng(42)
    n_users = 30
    n_movies = 20
    noise = 0.05
    B = rng.normal(loc=0, scale=0.5, size=((n_users, 2)))
    C = rng.normal(loc=0, scale=0.5, size=((n_movies, 2)))
    random_noise = rng.normal(loc=0, scale=noise, size=((30, 20)))
    R = B @ np.transpose(C) + random_noise

    users = np.repeat(np.arange(n_users), n_movies)
    movies = np.tile(np.arange(n_movies), n_users)
    values = np.ravel(R)

    perm = rng.permutation(len(users))
    users, movies, values = users[perm], movies[perm], values[perm]
    train = (users[0:300], movies[0:300], values[0:300])
    train_users, train_movies, train_values = train

    k, reg, scale, n_epochs = 2, 0.01, 0.1, 10

    model = init_factor_model(n_users=30, n_movies=20, k=k, scale=scale, rng=rng)
    user_positions = group_indices(ids=train_users, n=30)
    movie_positions = group_indices(ids=train_movies, n=20)

    for i in range(n_epochs):
        loss_before = als_loss(model=model, train=train, weighted=weighted, reg=reg)

        update_users(
            model=model,
            residuals=train_values,
            reg=reg,
            movies=train_movies,
            user_positions=user_positions,
            weighted=weighted,
        )
        loss_after_users = als_loss(
            model=model, train=train, weighted=weighted, reg=reg
        )
        assert loss_after_users <= loss_before + 1e-9

        update_movies(
            model=model,
            residuals=train_values,
            reg=reg,
            users=train_users,
            movie_positions=movie_positions,
            weighted=weighted,
        )
        loss_after_movies = als_loss(
            model=model, train=train, weighted=weighted, reg=reg
        )
        assert loss_after_movies <= loss_after_users + 1e-9
