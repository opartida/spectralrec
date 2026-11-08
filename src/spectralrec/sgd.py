from dataclasses import dataclass

import numpy as np

from spectralrec.lowrank import predict_observed
from spectralrec.metrics import rmse


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


@dataclass
class TrainingHistory:
    epoch: list
    train_rmse: list
    val_rmse: list
    best_epoch: int


def train_sgd(
    train, val, n_users, n_movies, k, lr, reg, scale, n_epochs, patience, seed
):
    users, movies, residuals = train
    users_val, movies_val, residuals_val = val
    rng = np.random.default_rng(seed)
    model = init_factor_model(n_users, n_movies, k, scale, rng)
    history = TrainingHistory(epoch=[], train_rmse=[], val_rmse=[], best_epoch=0)
    best_val_rmse = float("Inf")
    num_steps_stale = 0
    for epoch in range(n_epochs):
        for j in rng.permutation(len(residuals)):
            sgd_step(
                model=model,
                u=users[j],
                i=movies[j],
                residual=residuals[j],
                lr=lr,
                reg=reg,
            )
        history.train_rmse.append(
            rmse(
                y_true=residuals,
                y_pred=predict_observed(
                    user_factors=model.user_factors,
                    movie_factors=model.movie_factors,
                    users=users,
                    movies=movies,
                ),
            )
        )
        history.val_rmse.append(
            rmse(
                y_true=residuals_val,
                y_pred=predict_observed(
                    user_factors=model.user_factors,
                    movie_factors=model.movie_factors,
                    users=users_val,
                    movies=movies_val,
                ),
            )
        )
        history.epoch.append(epoch)

        if history.val_rmse[epoch] < best_val_rmse:
            best_val_rmse = history.val_rmse[epoch]
            best_user_factors = model.user_factors.copy()
            best_movie_factors = model.movie_factors.copy()
            num_steps_stale = 0
            history.best_epoch = epoch
        else:
            num_steps_stale += 1

        if num_steps_stale >= patience:
            break
    model.user_factors = best_user_factors
    model.movie_factors = best_movie_factors

    return model, history
