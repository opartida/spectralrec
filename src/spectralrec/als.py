import numpy as np

from spectralrec.lowrank import predict_observed
from spectralrec.metrics import rmse
from spectralrec.sgd import TrainingHistory, init_factor_model


def solve_ridge(F, r, reg) -> np.ndarray:
    A = np.transpose(F) @ F + reg * np.eye(N=F.shape[1])
    b = np.transpose(F) @ r
    return np.linalg.solve(A, b)


def group_indices(ids, n) -> list:
    ids = np.asarray(ids)
    positions = []
    for i in range(n):
        positions.append(np.flatnonzero(ids == i))
    return positions


def update_users(model, residuals, reg, movies, user_positions, weighted):
    k = model.user_factors.shape[1]
    n_users = len(user_positions)
    for u in np.arange(n_users):
        user_pos = user_positions[u]
        if weighted:
            weighted_reg = len(user_pos) * reg
        else:
            weighted_reg = reg
        F = model.movie_factors[movies[user_pos]]
        r = residuals[user_pos]
        if len(user_pos) == 0:
            model.user_factors[u] = np.zeros(k)
        else:
            model.user_factors[u] = solve_ridge(F=F, r=r, reg=weighted_reg)


def update_movies(model, residuals, reg, users, movie_positions, weighted):
    k = model.movie_factors.shape[1]
    n_movies = len(movie_positions)
    for v in np.arange(n_movies):
        movie_pos = movie_positions[v]
        if weighted:
            weighted_reg = len(movie_pos) * reg
        else:
            weighted_reg = reg
        F = model.user_factors[users[movie_pos]]
        r = residuals[movie_pos]
        if len(movie_pos) == 0:
            model.movie_factors[v] = np.zeros(k)
        else:
            model.movie_factors[v] = solve_ridge(F=F, r=r, reg=weighted_reg)


def als_loss(model, train, weighted, reg):
    users, movies, residuals = train
    y_pred = predict_observed(
        user_factors=model.user_factors,
        movie_factors=model.movie_factors,
        users=users,
        movies=movies,
    )

    if weighted:
        return np.sum((residuals - y_pred) ** 2) + reg * np.sum(
            model.user_factors[users] * model.user_factors[users]
            + model.movie_factors[movies] * model.movie_factors[movies]
        )

    else:
        return np.sum((residuals - y_pred) ** 2) + reg * (
            np.sum(model.user_factors * model.user_factors)
            + np.sum(model.movie_factors * model.movie_factors)
        )


def train_als(
    train, val, n_users, n_movies, k, weighted, reg, scale, n_epochs, patience, seed
):
    users, movies, residuals = train
    users_val, movies_val, residuals_val = val
    rng = np.random.default_rng(seed)
    model = init_factor_model(n_users, n_movies, k, scale, rng)
    history = TrainingHistory(epoch=[], train_rmse=[], val_rmse=[], best_epoch=0)
    best_val_rmse = float("Inf")
    user_positions = group_indices(ids=users, n=n_users)
    movie_positions = group_indices(ids=movies, n=n_movies)
    num_steps_stale = 0
    for epoch in range(n_epochs):
        update_users(
            model=model,
            residuals=residuals,
            reg=reg,
            movies=movies,
            user_positions=user_positions,
            weighted=weighted,
        )

        update_movies(
            model=model,
            residuals=residuals,
            reg=reg,
            users=users,
            movie_positions=movie_positions,
            weighted=weighted,
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
