import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from spectralrec.lowrank import predict_observed
from spectralrec.metrics import mae, rmse
from spectralrec.pipeline import compute_residuals, prepare_data
from spectralrec.sgd import train_sgd


def plot_run(history, baseline_rmse):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("RMSE per epoch for training and validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    ax.plot(history.epoch, history.train_rmse, label="train")
    ax.plot(history.epoch, history.val_rmse, label="validation")
    ax.axhline(baseline_rmse, linestyle="--", label="bias baseline (val)", color="gray")
    ax.legend()
    fig.savefig("reports/figures/learning_curve_sgd.png")


def main():
    RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        sgd_config = json.load(f)

    with open("configs/baseline.json") as f:
        baseline_config = json.load(f)
    data = prepare_data(ratings_path=RATINGS_PATH, split_seed=sgd_config["split_seed"])
    residuals = compute_residuals(
        data, baseline_config["reg_movie"], baseline_config["reg_user"]
    )

    n_users = np.size(data.index_maps.user_ids)
    n_movies = np.size(data.index_maps.movie_ids)
    start = time.perf_counter()
    model, history = train_sgd(
        train=(data.train.users, data.train.movies, residuals.train_residuals),
        val=(data.val.users, data.val.movies, residuals.val_residuals),
        n_users=n_users,
        n_movies=n_movies,
        k=sgd_config["k"],
        lr=sgd_config["lr"],
        reg=sgd_config["reg"],
        scale=sgd_config["scale"],
        n_epochs=sgd_config["n_epochs"],
        patience=sgd_config["patience"],
        seed=sgd_config["model_seed"],
    )
    seconds = time.perf_counter() - start
    baseline_rmse = rmse(
        residuals.val_residuals, np.zeros_like(residuals.val_residuals)
    )
    print(f"seconds: {seconds:.2f}")
    print(f"time per epoch: {(seconds / len(history.epoch)):.2f}")
    print(f"best epoch: {history.best_epoch}")
    print(f"validation rmse: {history.val_rmse}")
    plot_run(history=history, baseline_rmse=baseline_rmse)

    val_mae = mae(
        residuals.val_residuals,
        predict_observed(
            user_factors=model.user_factors,
            movie_factors=model.movie_factors,
            users=data.val.users,
            movies=data.val.movies,
        ),
    )

    git_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    run = {
        "run_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "git_commit": git_commit,
        "model": sgd_config["model"],
        "split_seed": sgd_config["split_seed"],
        "model_seed": sgd_config["model_seed"],
        "lam": None,
        "weighted": None,
        "k": sgd_config["k"],
        "lr": sgd_config["lr"],
        "reg": sgd_config["reg"],
        "scale": sgd_config["scale"],
        "n_epochs": sgd_config["n_epochs"],
        "patience": sgd_config["patience"],
        "train_rmse": history.train_rmse[history.best_epoch],
        "val_rmse": history.val_rmse[history.best_epoch],
        "val_mae": val_mae,
        "best_epoch": history.best_epoch,
        "seconds": seconds,
    }

    Path("results").mkdir(parents=True, exist_ok=True)
    np.savez(
        "results/sgd.npz",
        user_factors=model.user_factors,
        movie_factors=model.movie_factors,
    )

    file_exists = Path("reports/experiments.csv").exists()
    with open("reports/experiments.csv", mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(run)


if __name__ == "__main__":
    main()
