import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectralrec.baselines import BiasBaseline, GlobalMean, UserMean
from spectralrec.config import SPLIT_SEED
from spectralrec.metrics import mae, rmse
from spectralrec.pipeline import prepare_data

RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")


def main():
    """Fit the baseline models on train, compare them on validation and save the results."""

    data = prepare_data(RATINGS_PATH, SPLIT_SEED)

    n_users = np.size(data.index_maps.user_ids)
    n_movies = np.size(data.index_maps.movie_ids)

    reg_users = (0, 0.5, 1, 3, 10)
    reg_movies = (0, 0.5, 1, 3, 10)
    results = []
    for reg_user, reg_movie in itertools.product(reg_users, reg_movies):
        model = BiasBaseline(
            n_users=n_users, n_movies=n_movies, reg_movie=reg_movie, reg_user=reg_user
        )

        model.fit(data.train.users, data.train.movies, data.train.ratings)
        prediction = model.predict(data.val.users, data.val.movies)
        model_rmse = rmse(data.val.ratings, prediction)

        results.append(
            {"reg_movie": reg_movie, "reg_user": reg_user, "val_rmse": model_rmse}
        )
    best_result = min(results, key=lambda row: row["val_rmse"])
    print(
        f"Best regularization parameters reg_user: {best_result['reg_user']}, reg_movie: {best_result['reg_movie']}, value: {best_result['val_rmse']}"
    )
    pd.DataFrame(data=results).to_csv("reports/bias_grid_val.csv", index=False)
    with open("configs/baseline.json", "w") as f:
        json.dump(
            {
                "reg_movie": best_result["reg_movie"],
                "reg_user": best_result["reg_user"],
            },
            f,
            indent=2,
        )

    models = [
        ("GlobalMean", GlobalMean(n_users=n_users, n_movies=n_movies)),
        ("UserMean", UserMean(n_users=n_users, n_movies=n_movies)),
        (
            "BiasBaseline",
            BiasBaseline(
                n_users=n_users,
                n_movies=n_movies,
                reg_movie=best_result["reg_movie"],
                reg_user=best_result["reg_user"],
            ),
        ),
    ]

    values = []
    for name, model in models:
        model.fit(data.train.users, data.train.movies, data.train.ratings)
        prediction = model.predict(data.val.users, data.val.movies)
        model_rmse = rmse(data.val.ratings, prediction)
        model_mae = mae(data.val.ratings, prediction)
        values.append({"name": name, "val_mae": model_mae, "val_rmse": model_rmse})

        print(f"{name} RMSE:  {model_rmse:.4f} MAE:  {model_mae:.4f}")

    pd.DataFrame(data=values).to_csv("reports/baselines_val.csv", index=False)


if __name__ == "__main__":
    main()
