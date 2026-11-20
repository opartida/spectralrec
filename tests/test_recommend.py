import numpy as np
import pandas as pd
import pytest

from spectralrec.baselines import BiasBaseline
from spectralrec.data import IndexMaps
from spectralrec.recommend import SoftImputeRecommender, recommend


def build_small_world():
    maps = IndexMaps(
        user_ids=np.array([3, 56]),
        movie_ids=np.array([67, 87, 98, 301]),
        user_to_index={3: 0, 56: 1},
        movie_to_index={67: 0, 87: 1, 98: 2, 301: 3},
    )

    bias_baseline = BiasBaseline(n_users=2, n_movies=4, reg_movie=10, reg_user=10)
    bias_baseline.global_mean = 3.5
    bias_baseline.movie_bias = np.array([0.2, 0.5, 0.2, 0.1])
    bias_baseline.user_bias = np.array([0.4, 0.8])
    model = SoftImputeRecommender(
        bias_model=bias_baseline,
        Z=np.array([[0.3, 0.5, 0.6, 0.1], [0.7, 0.1, 0.9, 1]]),
    )

    movies = pd.DataFrame(
        {
            "movieId": [67, 87, 98, 301],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D"],
        }
    )

    return model, maps, movies


def test_recommend_excludes_seen():
    model, maps, movies = build_small_world()
    seen_movie_ids = [87, 98]
    movie_scores = recommend(
        model=model, user_id=3, maps=maps, movies=movies, seen_movie_ids=seen_movie_ids
    )

    assert not movie_scores["movieId"].isin(seen_movie_ids).any()


def test_recommend_returns_n_sorted():
    model, maps, movies = build_small_world()
    seen_movie_ids = []
    movie_scores = recommend(
        model=model,
        user_id=56,
        maps=maps,
        movies=movies,
        seen_movie_ids=seen_movie_ids,
        n=2,
    )

    assert len(movie_scores) == 2
    assert (np.diff(movie_scores["score"].to_numpy()) <= 0).all()


def test_recommend_unknown_user():
    model, maps, movies = build_small_world()
    seen_movie_ids = []
    with pytest.raises(ValueError):
        recommend(
            model=model,
            user_id=90,
            maps=maps,
            movies=movies,
            seen_movie_ids=seen_movie_ids,
            n=2,
        )
