import argparse
import json
from pathlib import Path

import numpy as np

from spectralrec.baselines import BiasBaseline
from spectralrec.data import load_movies, load_ratings
from spectralrec.pipeline import EncodedRatings, prepare_data
from spectralrec.recommend import SoftImputeRecommender, recommend
from spectralrec.softimpute import soft_impute


def main():
    RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")
    MOVIES_PATH = Path("data/raw/ml-latest-small/movies.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--top-n", required=True, type=int)
    args = parser.parse_args()
    with open("configs/best.json") as f:
        run_config = json.load(f)

    with open("configs/baseline.json") as f:
        baseline_config = json.load(f)

    reg_movie = baseline_config["reg_movie"]
    reg_user = baseline_config["reg_user"]
    data = prepare_data(
        ratings_path=RATINGS_PATH, split_seed=run_config["sgd"]["split_seed"]
    )
    movies = load_movies(MOVIES_PATH)
    ratings = load_ratings(RATINGS_PATH)
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
    Z = U @ np.diag(s) @ Vt
    model = SoftImputeRecommender(bias_model=bias_baseline, Z=Z)
    seen_movies = ratings[ratings["userId"] == args.user_id]

    movie_scores = recommend(
        model=model,
        user_id=args.user_id,
        maps=data.index_maps,
        movies=movies,
        seen_movie_ids=seen_movies["movieId"],
        n=args.top_n,
    )

    print(movie_scores.to_string(index=False))


if __name__ == "__main__":
    main()
