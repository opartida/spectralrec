import numpy as np

from spectralrec.lowrank import compute_singular_values, to_dense
from spectralrec.softimpute import soft_impute, soft_threshold_svd


def test_soft_threshold_svd_diagonal():
    X = np.diag([5, 3, 1])
    lam = 2
    U, s, Vt = soft_threshold_svd(X, lam)
    np.testing.assert_array_almost_equal(s, [3, 1])

    X_reconstructed = U @ np.diag(s) @ Vt
    np.testing.assert_array_almost_equal(
        X_reconstructed, np.array([[3, 0, 0], [0, 1, 0], [0, 0, 0]])
    )


def test_soft_impute_returns_zero_when_lambda_is_sigma1():
    users = np.array([0, 2, 1, 0, 1])
    movies = np.array([0, 0, 0, 1, 1])
    residuals = np.array([0.2, 0.3, 0.4, 0.9, 0.1])
    shape = (3, 3)
    X = to_dense(users=users, movies=movies, values=residuals, shape=shape)
    sigma1 = compute_singular_values(matrix=X)[0]
    U, s, Vt = soft_impute(
        users=users,
        movies=movies,
        residuals=residuals,
        shape=shape,
        lam=sigma1,
        max_iter=10,
        tol=1e-3,
    )

    np.testing.assert_array_almost_equal(U @ np.diag(s) @ Vt, np.zeros(shape=shape))


def test_nuclear_norm_identity():
    rng = np.random.default_rng(42)
    shape = (5, 3)
    Z = rng.normal(loc=0, scale=1, size=shape)

    U, s, Vt = np.linalg.svd(a=Z, full_matrices=False, compute_uv=True)

    square_root_sigma = np.sqrt(np.diag(s))
    A = U @ square_root_sigma
    B = np.transpose(Vt) @ square_root_sigma

    assert abs((1 / 2) * (np.sum(A**2) + np.sum(B**2)) - np.sum(s)) < 1e-6
    np.testing.assert_almost_equal(A @ np.transpose(B), Z)
