from dataclasses import dataclass

import numpy as np


@dataclass
class FactorModel:
    user_factors: np.ndarray  # (n_users, k)
    movie_factors: np.ndarray  # (n_movies, k)


def init_factor_model(n_users, n_movies, k, scale, rng) -> FactorModel:
    return FactorModel(
        user_factors=rng.normal(loc=0, scale=scale, size=(n_users, k)),
        movie_factors=rng.normal(loc=0, scale=scale, size=(n_movies, k)),
    )


def sgd_step(model, u, i, residual, lr, reg) -> float:
    prediction = model.user_factors[u] @ model.movie_factors[i]
    error = residual - prediction
    if not np.isfinite(error) or abs(error) > 30:
        raise FloatingPointError
    old_user = model.user_factors[u].copy()
    model.user_factors[u] += lr * (
        error * model.movie_factors[i] - reg * model.user_factors[u]
    )

    model.movie_factors[i] += lr * (
        (residual - prediction) * old_user - reg * model.movie_factors[i]
    )

    return error
