import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectralrec.config import RESIDUALS_PERMUTATION_SEED, SPLIT_SEED
from spectralrec.lowrank import compute_singular_values, to_dense
from spectralrec.pipeline import compute_residuals, prepare_data

RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")


def plot_singular_values(singular_values, control_singular_values, fig_title, fig_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(fig_title)
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
    fig.savefig(fig_path)


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

    singular_values = compute_singular_values(matrix=residual_matrix)

    rng = np.random.default_rng(RESIDUALS_PERMUTATION_SEED)

    list_control_first_singular_value = []
    for _ in range(20):
        residual_control_matrix = to_dense(
            data.train.users,
            data.train.movies,
            rng.permutation(residuals.train_residuals),
            shape=shape,
        )

        control_singular_values = compute_singular_values(
            matrix=residual_control_matrix
        )

        list_control_first_singular_value.append(control_singular_values[0])

    print(f"First singular value: {singular_values[0]:.2f}")
    print(
        f"Max control first singular value: {np.max(list_control_first_singular_value):.2f}"
    )
    print(
        f"Min control first singular value: {np.min(list_control_first_singular_value):.2f}"
    )

    plot_singular_values(
        singular_values=singular_values,
        control_singular_values=control_singular_values,
        fig_title="Full matrix",
        fig_path="reports/figures/spectrum.png",
    )

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
    dense_singular_values = compute_singular_values(matrix=dense_submatrix)

    mask = dense_submatrix != 0
    dense_control = dense_submatrix.copy()

    list_control_dense_first_singular_value = []
    for _ in range(20):
        dense_control[mask] = rng.permutation(dense_submatrix[mask])

        dense_control_singular_values = compute_singular_values(matrix=dense_control)

        list_control_dense_first_singular_value.append(dense_control_singular_values[0])

    print(f"First dense singular value: {dense_singular_values[0]:.2f}")
    print(
        f"Max dense control first singular value: {np.max(list_control_dense_first_singular_value):.2f}"
    )
    print(
        f"Min dense control first singular value: {np.min(list_control_dense_first_singular_value):.2f}"
    )

    plot_singular_values(
        singular_values=dense_singular_values,
        control_singular_values=dense_control_singular_values,
        fig_title=f"Dense {best_size_sub_matrix} × {best_size_sub_matrix} submatrix",
        fig_path="reports/figures/spectrum_dense.png",
    )

    print(f"First 10 singular values, dense submatrix: {dense_singular_values[:10]}")
    print(
        f"First 10 singular values, shuffled control: {dense_control_singular_values[:10]}"
    )


if __name__ == "__main__":
    main()
