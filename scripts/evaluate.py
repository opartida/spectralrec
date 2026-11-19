import argparse
import json
from pathlib import Path

import numpy as np

from spectralrec.als import group_indices, update_movies, update_users
from spectralrec.baselines import BiasBaseline, GlobalMean
from spectralrec.lowrank import TruncatedSVD, predict_observed
from spectralrec.metrics import mae, rmse
from spectralrec.pipeline import EncodedRatings, prepare_data
from spectralrec.sgd import init_factor_model, sgd_step
from spectralrec.softimpute import soft_impute


def main():
    RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["val", "test"])
    args = parser.parse_args()
    with open("configs/best.json") as f:
        run_config = json.load(f)

    with open("configs/baseline.json") as f:
        baseline_config = json.load(f)

    data = prepare_data(
        ratings_path=RATINGS_PATH, split_seed=run_config["sgd"]["split_seed"]
    )
    n_users = np.size(data.index_maps.user_ids)
    n_movies = np.size(data.index_maps.movie_ids)
    if args.split == "val":
        fit_data = data.train
        eval_data = data.val
    elif args.split == "test":
        fit_data = EncodedRatings(
            users=np.concatenate([data.train.users, data.val.users]),
            movies=np.concatenate([data.train.movies, data.val.movies]),
            ratings=np.concatenate([data.train.ratings, data.val.ratings]),
        )
        eval_data = data.test

    global_mean = GlobalMean(n_users=n_users, n_movies=n_movies)
    global_mean.fit(
        users=fit_data.users, movies=fit_data.movies, ratings=fit_data.ratings
    )
    global_mean_predictions = global_mean.predict(
        users=eval_data.users, movies=eval_data.movies
    )

    reg_movie = baseline_config["reg_movie"]
    reg_user = baseline_config["reg_user"]
    bias_baseline = BiasBaseline(
        n_users=n_users, n_movies=n_movies, reg_movie=reg_movie, reg_user=reg_user
    )
    bias_baseline.fit(
        users=fit_data.users, movies=fit_data.movies, ratings=fit_data.ratings
    )
    bias_baseline_predictions = bias_baseline.predict(
        users=eval_data.users, movies=eval_data.movies
    )
    fit_residuals = fit_data.ratings - bias_baseline.predict(
        fit_data.users, fit_data.movies
    )
    eval_residuals = eval_data.ratings - bias_baseline_predictions

    results = []
    global_mean_rmse = rmse(eval_data.ratings, global_mean_predictions)
    bias_baseline_rmse = rmse(eval_data.ratings, bias_baseline_predictions)
    global_mean_mae = mae(eval_data.ratings, global_mean_predictions)
    bias_baseline_mae = mae(eval_data.ratings, bias_baseline_predictions)

    results.append(("global_mean", global_mean_rmse, global_mean_mae))
    results.append(("bias_baseline", bias_baseline_rmse, bias_baseline_mae))
    sgd_predictions = []
    diffs = []
    for model_name in run_config.keys():
        config = run_config[model_name]

        if model_name == "als":
            k = config["k"]
            scale = config["scale"]
            epochs = config["epochs"]
            reg = config["reg"]
            weighted = config["weighted"]
            rng = np.random.default_rng(config["model_seed"])
            als_model = init_factor_model(n_users, n_movies, k, scale, rng)
            user_positions = group_indices(ids=fit_data.users, n=n_users)
            movie_positions = group_indices(ids=fit_data.movies, n=n_movies)

            for epoch in range(epochs):
                update_users(
                    model=als_model,
                    residuals=fit_residuals,
                    reg=reg,
                    movies=fit_data.movies,
                    user_positions=user_positions,
                    weighted=weighted,
                )

                update_movies(
                    model=als_model,
                    residuals=fit_residuals,
                    reg=reg,
                    users=fit_data.users,
                    movie_positions=movie_positions,
                    weighted=weighted,
                )

            y_pred = (
                predict_observed(
                    user_factors=als_model.user_factors,
                    movie_factors=als_model.movie_factors,
                    users=eval_data.users,
                    movies=eval_data.movies,
                )
                + bias_baseline_predictions
            )
            als_rmse = rmse(eval_data.ratings, y_pred)
            als_mae = mae(eval_data.ratings, y_pred)
            results.append(("als", als_rmse, als_mae))
        elif model_name == "sgd":
            k = config["k"]
            scale = config["scale"]
            epochs = config["epochs"]
            lr = config["lr"]
            reg = config["reg"]
            rng = np.random.default_rng(config["model_seed"])
            sgd_model = init_factor_model(n_users, n_movies, k, scale, rng)
            for epoch in range(epochs):
                for j in rng.permutation(len(fit_residuals)):
                    sgd_step(
                        model=sgd_model,
                        u=fit_data.users[j],
                        i=fit_data.movies[j],
                        residual=fit_residuals[j],
                        lr=lr,
                        reg=reg,
                    )
            y_pred = (
                predict_observed(
                    user_factors=sgd_model.user_factors,
                    movie_factors=sgd_model.movie_factors,
                    users=eval_data.users,
                    movies=eval_data.movies,
                )
                + bias_baseline_predictions
            )
            sgd_predictions = y_pred
            sgd_rmse = rmse(eval_data.ratings, y_pred)
            sgd_mae = mae(eval_data.ratings, y_pred)
            results.append(("sgd", sgd_rmse, sgd_mae))
        elif model_name == "svd":
            truncated_svd = TruncatedSVD(k=config["k"])
            truncated_svd.fit(
                users=fit_data.users,
                movies=fit_data.movies,
                residuals=fit_residuals,
                shape=(n_users, n_movies),
            )
            truncated_svd_predictions = truncated_svd.predict(
                users=eval_data.users, movies=eval_data.movies
            )
            truncated_svd_predictions = (
                truncated_svd_predictions + bias_baseline_predictions
            )
            truncated_svd_rmse = rmse(eval_data.ratings, truncated_svd_predictions)
            truncated_svd_mae = mae(eval_data.ratings, truncated_svd_predictions)
            results.append(("truncated_svd", truncated_svd_rmse, truncated_svd_mae))
        elif model_name == "soft":
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
            soft_impute_predictions = (
                Z[eval_data.users, eval_data.movies] + bias_baseline_predictions
            )
            soft_impute_rmse = rmse(eval_data.ratings, soft_impute_predictions)
            soft_impute_mae = mae(eval_data.ratings, soft_impute_predictions)
            results.append(("soft_impute", soft_impute_rmse, soft_impute_mae))

        else:
            raise ValueError(f"Incorrect model name: {model_name}")

    print(f"| {'model':>13} | {'RMSE ' + args.split:>10} | {'MAE ' + args.split:>10} |")
    print(f"|{'-' * 15}|{'-' * 12}|{'-' * 12}|")

    for model_name, rmse_value, mae_value in results:
        print(f"| {model_name:>13} | {rmse_value:>10.4f} | {mae_value:>10.4f} |")

    if args.split == "test":
        rng = np.random.default_rng(0)
        n = len(eval_data.ratings)
        diff = bias_baseline_rmse - sgd_rmse
        for i in range(1000):
            idx = rng.integers(0, n, size=n)

            boot_bias_rmse = rmse(
                eval_data.ratings[idx], bias_baseline_predictions[idx]
            )

            boot_sgd_rmse = rmse(eval_data.ratings[idx], sgd_predictions[idx])

            diffs.append(boot_bias_rmse - boot_sgd_rmse)

        low, high = np.percentile(diffs, [2.5, 97.5])
        print(f"Interval low-high: [{low:.4f}, {high:.4f}]")
        print(f"Diff RMSE baseline - RMSE SGD: {diff:.4f}")


if __name__ == "__main__":
    main()
