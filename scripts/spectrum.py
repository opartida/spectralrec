import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectralrec.config import RESIDUALS_PERMUTATION_SEED, SPLIT_SEED
from spectralrec.lowrank import to_dense
from spectralrec.pipeline import compute_residuals, prepare_data

RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")


def main():
    """Compare the singular values of the train residuals with a shuffled control and save the figure."""

    with open("configs/baseline.json") as f:
        config = json.load(f)
    data = prepare_data(ratings_path=RATINGS_PATH, split_seed=SPLIT_SEED)
    shape = (len(data.index_maps.user_ids), len(data.index_maps.movie_ids))
    residuals = compute_residuals(data, config["reg_movie"], config["reg_user"])
    residual_matrix = to_dense(
        data.train.users,
        data.train.movies,
        residuals.train_residuals,
        shape=shape,
    )

    singular_values = np.linalg.svd(
        residual_matrix, full_matrices=False, compute_uv=False
    )

    rng = np.random.default_rng(RESIDUALS_PERMUTATION_SEED)
    residual_control_matrix = to_dense(
        data.train.users,
        data.train.movies,
        rng.permutation(residuals.train_residuals),
        shape=shape,
    )

    control_singular_values = np.linalg.svd(
        residual_control_matrix, full_matrices=False, compute_uv=False
    )

    print(f"Singular values residual matrix: {singular_values}")
    print(f"Singular values control matrix: {control_singular_values}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Singular values")
    ax1.set_yscale("log")

    x_values = np.arange(1, len(singular_values) + 1)
    ax1.set_xlabel("j")
    ax1.set_ylabel(r"$\sigma_j$")
    ax1.plot(x_values, singular_values, label="train residuals")
    ax1.plot(
        x_values,
        control_singular_values,
        label="shuffled control",
    )
    ax1.legend()

    ax2.set_title("Cumulative energy")
    ax2.set_xlabel("k")
    ax2.set_ylabel(r"$\sum_{j\leq k}\sigma_j^2/\sum_{j}\sigma_j^2$")
    ax2.plot(
        x_values,
        np.cumsum(singular_values**2) / np.sum(singular_values**2),
        label="train residuals",
    )
    ax2.plot(
        x_values,
        np.cumsum(control_singular_values**2) / np.sum(control_singular_values**2),
        label="shuffled control",
    )
    ax2.legend()
    fig.savefig("reports/figures/spectrum.png")

    ratings_per_movie = np.bincount(
        data.train.movies, minlength=len(data.index_maps.movie_ids)
    )
    ratings_per_user = np.bincount(
        data.train.users, minlength=len(data.index_maps.user_ids)
    )

    sorted_user_idx = np.argsort(ratings_per_user)
    sorted_movie_idx = np.argsort(ratings_per_movie)
    for val in [50, 100, 200]:
        dense_submatrix = residual_matrix[
            np.ix_(sorted_user_idx[-val:], sorted_movie_idx[-val:])
        ]
        size = np.size(dense_submatrix)
        density = np.count_nonzero(dense_submatrix) / size
        print(f"Density for {dense_submatrix.shape}: {density:.1%}")

    best_size_sub_matrix = 50
    dense_submatrix = residual_matrix[
        np.ix_(
            sorted_user_idx[-best_size_sub_matrix:],
            sorted_movie_idx[-best_size_sub_matrix:],
        )
    ]
    dense_singular_values = np.linalg.svd(
        dense_submatrix, full_matrices=False, compute_uv=False
    )

    mask = dense_submatrix != 0
    dense_control = dense_submatrix.copy()
    dense_control[mask] = rng.permutation(dense_submatrix[mask])

    dense_control_singular_values = np.linalg.svd(
        dense_control, full_matrices=False, compute_uv=False
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Singular values")
    ax1.set_yscale("log")

    x_values = np.arange(1, len(dense_singular_values) + 1)
    ax1.set_xlabel("j")
    ax1.set_ylabel(r"$\sigma_j$")
    ax1.plot(x_values, dense_singular_values, label="train residuals")
    ax1.plot(
        x_values,
        dense_control_singular_values,
        label="shuffled control",
    )
    ax1.legend()

    fig.savefig("reports/figures/spectrum_dense.png")


if __name__ == "__main__":
    main()
