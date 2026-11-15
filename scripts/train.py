import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from spectralrec.als import train_als
from spectralrec.lowrank import predict_observed
from spectralrec.metrics import mae, rmse
from spectralrec.pipeline import compute_residuals, prepare_data
from spectralrec.sgd import train_sgd


def plot_run(history, baseline_rmse, run_config, run_id):
    model_name = run_config["model"]
    k = run_config["k"]
    if model_name == "sgd":
        extra = f"lr = {run_config['lr']}"
    else:
        extra = f"weighted = {run_config['weighted']}"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(
        f"{model_name.upper()} (k = {k}, reg = {run_config['reg']}, {extra}): "
        "RMSE per epoch"
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    ax.plot(history.epoch, history.train_rmse, label="train")
    ax.plot(history.epoch, history.val_rmse, label="validation")
    ax.axhline(baseline_rmse, linestyle="--", label="bias baseline (val)", color="gray")
    ax.legend()
    fig.savefig(
        f"results/figures/learning_curve_{run_id}_{model_name}_k{k}_reg{run_config['reg']}.png"
    )


def main():
    RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        run_config = json.load(f)

    with open("configs/baseline.json") as f:
        baseline_config = json.load(f)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    data = prepare_data(ratings_path=RATINGS_PATH, split_seed=run_config["split_seed"])
    residuals = compute_residuals(
        data, baseline_config["reg_movie"], baseline_config["reg_user"]
    )

    n_users = np.size(data.index_maps.user_ids)
    n_movies = np.size(data.index_maps.movie_ids)
    start = time.perf_counter()
    model_name = run_config["model"]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if model_name == "sgd":
        model, history = train_sgd(
            train=(data.train.users, data.train.movies, residuals.train_residuals),
            val=(data.val.users, data.val.movies, residuals.val_residuals),
            n_users=n_users,
            n_movies=n_movies,
            k=run_config["k"],
            lr=run_config["lr"],
            reg=run_config["reg"],
            scale=run_config["scale"],
            n_epochs=run_config["n_epochs"],
            patience=run_config["patience"],
            seed=run_config["model_seed"],
        )
    elif model_name == "als":
        model, history = train_als(
            train=(data.train.users, data.train.movies, residuals.train_residuals),
            val=(data.val.users, data.val.movies, residuals.val_residuals),
            n_users=n_users,
            n_movies=n_movies,
            k=run_config["k"],
            weighted=run_config["weighted"],
            reg=run_config["reg"],
            scale=run_config["scale"],
            n_epochs=run_config["n_epochs"],
            patience=run_config["patience"],
            seed=run_config["model_seed"],
        )
    else:
        raise ValueError(
            f"Incorrect model name: {model_name}, please use an allowed model name: als, sgd"
        )
    seconds = time.perf_counter() - start
    baseline_rmse = rmse(
        residuals.val_residuals, np.zeros_like(residuals.val_residuals)
    )
    print(f"seconds: {seconds:.2f}")
    print(f"time per epoch: {(seconds / len(history.epoch)):.2f}")
    print(f"best epoch: {history.best_epoch}")
    print(f"validation rmse: {history.val_rmse}")
    plot_run(
        history=history,
        baseline_rmse=baseline_rmse,
        run_config=run_config,
        run_id=run_id,
    )

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
        "run_id": run_id,
        "git_commit": git_commit,
        "model": run_config.get("model"),
        "split_seed": run_config.get("split_seed"),
        "model_seed": run_config.get("model_seed"),
        "lam": None,
        "weighted": run_config.get("weighted"),
        "k": run_config.get("k"),
        "lr": run_config.get("lr"),
        "reg": run_config.get("reg"),
        "scale": run_config.get("scale"),
        "n_epochs": run_config.get("n_epochs"),
        "patience": run_config.get("patience"),
        "train_rmse": history.train_rmse[history.best_epoch],
        "val_rmse": history.val_rmse[history.best_epoch],
        "val_mae": val_mae,
        "best_epoch": history.best_epoch,
        "seconds": seconds,
    }

    np.savez(
        f"results/{run_id}_{model_name}_k{run_config['k']}_reg{run_config['reg']}.npz",
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
