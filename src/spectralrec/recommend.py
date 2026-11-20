from dataclasses import dataclass

import numpy as np
import pandas as pd

from spectralrec.baselines import BiasBaseline
from spectralrec.data import IndexMaps


@dataclass
class SoftImputeRecommender:
    bias_model: BiasBaseline
    Z: np.ndarray


def recommend(
    model: SoftImputeRecommender, user_id, maps: IndexMaps, movies, seen_movie_ids, n=10
) -> pd.DataFrame:
    if user_id not in maps.user_to_index:
        raise ValueError(f"Unknown user_id: {user_id}")
    u = maps.user_to_index[user_id]
    n_movies = len(maps.movie_ids)

    user_ratings = (
        model.bias_model.predict(
            users=np.repeat(a=u, repeats=n_movies), movies=np.arange(n_movies)
        )
        + model.Z[u]
    )

    for seen_movie in seen_movie_ids:
        if seen_movie in maps.movie_to_index:
            user_ratings[maps.movie_to_index[seen_movie]] = -np.inf

    num_kept_movies = min(n, np.sum(np.asarray(user_ratings) != -np.inf))
    sorted_idx = np.argsort(-user_ratings)
    movie_ids = maps.movie_ids[sorted_idx][:num_kept_movies]
    movies = movies.set_index("movieId")
    titles = np.asarray(movies.loc[movie_ids, "title"])
    scores = user_ratings[sorted_idx][:num_kept_movies]

    return pd.DataFrame({"movieId": movie_ids, "title": titles, "score": scores})
