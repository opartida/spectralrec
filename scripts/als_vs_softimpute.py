import numpy as np

from spectralrec.als import (
    als_loss,
    group_indices,
    update_movies,
    update_users,
)
from spectralrec.sgd import init_factor_model
from spectralrec.softimpute import soft_impute


def soft_loss(U, s, Vt, train, reg):
    users, movies, residuals = train
    Z = U @ np.diag(s) @ Vt
    return (1 / 2) * np.sum((Z[users, movies] - residuals) ** 2) + reg * np.sum(s)


def main():
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

    k, reg, scale, n_epochs = 20, 0.01, 0.1, 600
    model = init_factor_model(
        n_users=n_users, n_movies=n_movies, k=k, scale=scale, rng=rng
    )
    user_positions = group_indices(ids=train_users, n=30)
    movie_positions = group_indices(ids=train_movies, n=20)

    for i in range(n_epochs):
        update_users(
            model=model,
            residuals=train_values,
            reg=reg,
            movies=train_movies,
            user_positions=user_positions,
            weighted=False,
        )

        update_movies(
            model=model,
            residuals=train_values,
            reg=reg,
            users=train_users,
            movie_positions=movie_positions,
            weighted=False,
        )
    computed_als_loss = als_loss(model=model, train=train, weighted=False, reg=reg)
    print(f"als loss: {computed_als_loss}")

    lam, max_iter, tol = reg, 10000, 1e-18

    U, s, Vt = soft_impute(
        users=train_users,
        movies=train_movies,
        residuals=train_values,
        shape=(n_users, n_movies),
        lam=lam,
        max_iter=max_iter,
        tol=tol,
    )

    computed_soft_loss = soft_loss(U=U, s=s, Vt=Vt, train=train, reg=reg)
    print(f"Soft loss: {computed_soft_loss}")


if __name__ == "__main__":
    main()
