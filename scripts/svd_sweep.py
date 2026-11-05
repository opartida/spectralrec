import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectralrec.config import SPLIT_SEED
from spectralrec.lowrank import TruncatedSVD
from spectralrec.metrics import rmse
from spectralrec.pipeline import compute_residuals, prepare_data

RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")


def plot_rmse(ks, rmse_train, rmse_val, baseline_rmse, fig_path):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_title("Truncated SVD: RMSE vs k")
    ax1.set_xscale("log")
    x_values = ks
    ax1.set_xlabel("k")
    ax1.set_ylabel("RMSE")
    ax1.plot(x_values, rmse_train, label="RMSE train", marker="x")
    ax1.plot(x_values, rmse_val, label="RMSE validation", marker="o")
    ax1.axhline(
        baseline_rmse, linestyle="--", label="bias baseline (val)", color="gray"
    )

    ax1.legend()

    fig.savefig(fig_path)


def main():
    with open("configs/baseline.json") as f:
        config = json.load(f)
    data = prepare_data(RATINGS_PATH, SPLIT_SEED)
    residuals = compute_residuals(data, config["reg_movie"], config["reg_user"])

    n_users = np.size(data.index_maps.user_ids)
    n_movies = np.size(data.index_maps.movie_ids)
    rmse_val = []
    rmse_train = []

    model = TruncatedSVD(k=100)

    model.fit(
        users=data.train.users,
        movies=data.train.movies,
        residuals=residuals.train_residuals,
        shape=(n_users, n_movies),
    )
    baseline_rmse = rmse(
        residuals.val_residuals, np.zeros_like(residuals.val_residuals)
    )

    ks = [1, 2, 5, 10, 20, 50, 100]
    for k in ks:
        model.truncate(k)
        pred_val = model.predict(users=data.val.users, movies=data.val.movies)
        pred_train = model.predict(users=data.train.users, movies=data.train.movies)
        rmse_val.append(rmse(residuals.val_residuals, pred_val))
        rmse_train.append(rmse(residuals.train_residuals, pred_train))

    print(f"| {'k':>3} | {'RMSE train':>10} | {'RMSE val':>8} |")

    print(f"|{'-' * 5}|{'-' * 12}|{'-' * 10}|")
    for k, train_rmse, val_rmse in zip(ks, rmse_train, rmse_val):
        print(f"| {k:>3} | {train_rmse:>10.4f} | {val_rmse:>8.4f} |")

    print(f"Bias baseline (val RMSE): {baseline_rmse:.4f}")

    plot_rmse(
        ks=ks,
        rmse_train=rmse_train,
        rmse_val=rmse_val,
        baseline_rmse=baseline_rmse,
        fig_path="reports/figures/svd_k.png",
    )


if __name__ == "__main__":
    main()
