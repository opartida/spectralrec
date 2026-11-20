import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectralrec.baselines import BiasBaseline
from spectralrec.data import load_movies
from spectralrec.pipeline import EncodedRatings, prepare_data
from spectralrec.softimpute import soft_impute


def plot_movies_map(Vt, ratings_per_movie, maps, movies, N, fig_path):
    movies = movies.set_index("movieId")
    fig, ax = plt.subplots(figsize=(12, 8))
    top = np.argsort(ratings_per_movie)[-N:]
    factor1 = Vt[0, :]
    factor2 = Vt[1, :]
    x, y = factor1[top], factor2[top]
    ax.set_xlabel("Axis 1: acclaimed / cult  ←→  blockbuster action")
    ax.set_ylabel("Axis 2: adult  ←→  family")
    ax.set_title("Most-rated movies on the first two Soft-Impute directions")

    ax.scatter(x, y)
    for idx in top[-15:]:
        x = factor1[idx]
        y = factor2[idx]
        title = movies.loc[maps.movie_ids[idx], "title"]
        ax.text(x, y, title, va="center", fontsize=8)

    fig.savefig(fig_path)


def get_movies_in_factor_ends(Vt, factor_idx, has_enough_ratings, movies, maps):
    factor = Vt[factor_idx]
    candidates_indices = np.where(has_enough_ratings)[0]
    sorted_idx = candidates_indices[np.argsort(factor[candidates_indices])]
    high_end = sorted_idx[-10:][::-1]
    low_end = sorted_idx[0:10]
    movies = movies.set_index("movieId")
    high_end_movie_titles = np.asarray(movies.loc[maps.movie_ids[high_end], "title"])
    low_end_movies_titles = np.asarray(movies.loc[maps.movie_ids[low_end], "title"])
    return high_end_movie_titles, low_end_movies_titles


def main():
    MIN_NUM_RATINGS = 50
    RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")
    MOVIES_PATH = Path("data/raw/ml-latest-small/movies.csv")

    movies = load_movies(MOVIES_PATH)

    with open("configs/best.json") as f:
        run_config = json.load(f)

    with open("configs/baseline.json") as f:
        baseline_config = json.load(f)

    reg_movie = baseline_config["reg_movie"]
    reg_user = baseline_config["reg_user"]
    data = prepare_data(
        ratings_path=RATINGS_PATH, split_seed=run_config["sgd"]["split_seed"]
    )

    n_users = np.size(data.index_maps.user_ids)
    n_movies = np.size(data.index_maps.movie_ids)
    fit_data = EncodedRatings(
        users=np.concatenate([data.train.users, data.val.users, data.test.users]),
        movies=np.concatenate([data.train.movies, data.val.movies, data.test.movies]),
        ratings=np.concatenate(
            [data.train.ratings, data.val.ratings, data.test.ratings]
        ),
    )
    bias_baseline = BiasBaseline(
        n_users=n_users, n_movies=n_movies, reg_movie=reg_movie, reg_user=reg_user
    )
    bias_baseline.fit(
        users=fit_data.users, movies=fit_data.movies, ratings=fit_data.ratings
    )
    fit_residuals = fit_data.ratings - bias_baseline.predict(
        fit_data.users, fit_data.movies
    )
    config = run_config["soft"]
    lam = config["lam"]
    max_iter = config["max_iter"]
    tol = config["tol"]
    U, s, Vt = soft_impute(
        users=fit_data.users,
        movies=fit_data.movies,
        residuals=fit_residuals,
        shape=(n_users, n_movies),
        lam=lam,
        max_iter=max_iter,
        tol=tol,
    )

    ratings_per_movie = np.bincount(fit_data.movies, minlength=n_movies)
    has_enough_ratings = ratings_per_movie >= MIN_NUM_RATINGS
    factor1_high_end_movie_titles, factor_1_low_end_movies_titles = (
        get_movies_in_factor_ends(
            Vt=Vt,
            factor_idx=0,
            has_enough_ratings=has_enough_ratings,
            movies=movies,
            maps=data.index_maps,
        )
    )

    factor2_high_end_movie_titles, factor_2_low_end_movies_titles = (
        get_movies_in_factor_ends(
            Vt=Vt,
            factor_idx=1,
            has_enough_ratings=has_enough_ratings,
            movies=movies,
            maps=data.index_maps,
        )
    )

    print("Factor 1, high end")
    print(factor1_high_end_movie_titles)
    print("Factor 1, low end")
    print(factor_1_low_end_movies_titles)
    print("Factor 2, high end")
    print(factor2_high_end_movie_titles)
    print("Factor 2, low end")
    print(factor_2_low_end_movies_titles)

    fig_path = "reports/figures/movie_map.png"
    plot_movies_map(
        Vt=Vt,
        ratings_per_movie=ratings_per_movie,
        maps=data.index_maps,
        movies=movies,
        N=100,
        fig_path=fig_path,
    )


if __name__ == "__main__":
    main()
