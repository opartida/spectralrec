import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectralrec.lowrank import compute_singular_values, to_dense
from spectralrec.metrics import rmse
from spectralrec.pipeline import compute_residuals, prepare_data
from spectralrec.softimpute import soft_impute

RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")


def plot_rmse_and_rank(lams, rmse_train, rmse_val, baseline_rmse, rank, fig_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title(r"Soft-Impute: RMSE vs $\lambda$")
    ax1.set_xlabel(r"$\lambda$")

    ax1.set_xscale("log")
    x_values = lams
    ax1.set_ylabel("RMSE")
    ax1.plot(x_values, rmse_train, label="RMSE train", marker="x")
    ax1.plot(x_values, rmse_val, label="RMSE validation", marker="o")
    ax1.axhline(
        baseline_rmse, linestyle="--", label="bias baseline (val)", color="gray"
    )

    ax2.set_xscale("log")
    ax2.set_xlabel("Lambda")
    ax2.set_ylabel("Rank")
    ax2.plot(x_values, rank, label="Rank")

    ax1.legend()
    ax2.legend()

    fig.savefig(fig_path)


def main():
    with open("configs/baseline.json") as f:
        config = json.load(f)

    with open("configs/soft.json") as f:
        soft_config = json.load(f)
    data = prepare_data(RATINGS_PATH, soft_config["split_seed"])
    residuals = compute_residuals(data, config["reg_movie"], config["reg_user"])

    n_users = np.size(data.index_maps.user_ids)
    n_movies = np.size(data.index_maps.movie_ids)
    rmse_val = []
    rmse_train = []
    rank = []
    seconds = []

    R = to_dense(
        users=data.train.users,
        movies=data.train.movies,
        values=residuals.train_residuals,
        shape=(n_users, n_movies),
    )

    baseline_rmse = rmse(
        residuals.val_residuals, np.zeros_like(residuals.val_residuals)
    )

    time_before_svd = time.perf_counter()
    np.linalg.svd(a=R, full_matrices=False, compute_uv=True)
    time_after_svd = time.perf_counter()
    print(f"time for svd: {time_after_svd - time_before_svd}")
    singular_values = compute_singular_values(matrix=R)
    previous_solution = None
    lams = np.geomspace(singular_values[0], singular_values[0] / 100, 10)
    for lam in lams:
        time_before_softimpute = time.perf_counter()
        U, s, Vt = soft_impute(
            users=data.train.users,
            movies=data.train.movies,
            residuals=residuals.train_residuals,
            shape=(n_users, n_movies),
            lam=lam,
            max_iter=soft_config["max_iter"],
            tol=soft_config["tol"],
            warm_start=previous_solution,
        )
        time_after_softimpute = time.perf_counter()
        seconds.append(time_after_softimpute - time_before_softimpute)

        print(f"time for softimpute: {time_after_softimpute - time_before_softimpute}")

        Z = U @ np.diag(s) @ Vt
        rmse_val.append(
            rmse(residuals.val_residuals, Z[data.val.users, data.val.movies])
        )

        rmse_train.append(
            rmse(residuals.train_residuals, Z[data.train.users, data.train.movies])
        )

        rank.append(len(s))

        previous_solution = (U, s, Vt)

    print(
        f"| {'λ':>8} | {'rank':>4} | {'RMSE train':>10} | {'RMSE val':>8} | {'seconds':>7} |"
    )
    print(f"|{'-' * 10}|{'-' * 6}|{'-' * 12}|{'-' * 10}|{'-' * 9}|")
    for lam, r, train, val, sec in zip(lams, rank, rmse_train, rmse_val, seconds):
        print(f"| {lam:>8.3f} | {r:>4} | {train:>10.4f} | {val:>8.4f} | {sec:>7.1f} |")

    plot_rmse_and_rank(
        lams=lams,
        rmse_train=rmse_train,
        rmse_val=rmse_val,
        baseline_rmse=baseline_rmse,
        rank=rank,
        fig_path="reports/figures/softimpute_path.png",
    )


if __name__ == "__main__":
    main()
