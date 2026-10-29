from dataclasses import dataclass

import numpy as np

from spectralrec.baselines import BiasBaseline
from spectralrec.data import (
    IndexMaps,
    build_index_maps,
    drop_unknown,
    encode_ratings,
    load_ratings,
    split_ratings,
    validate_ratings,
)


@dataclass
class EncodedRatings:
    users: np.ndarray
    movies: np.ndarray
    ratings: np.ndarray


@dataclass
class PreparedData:
    train: EncodedRatings
    val: EncodedRatings
    test: EncodedRatings
    index_maps: IndexMaps
    val_num_rows_deleted: int
    test_num_rows_deleted: int


@dataclass
class BiasBaselineResiduals:
    train_residuals: np.ndarray
    val_residuals: np.ndarray
    test_residuals: np.ndarray
    bias_model: BiasBaseline


def prepare_data(ratings_path, split_seed) -> PreparedData:
    data_ratings = load_ratings(ratings_path)
    validate_ratings(data_ratings)
    (train, val, test) = split_ratings(data_ratings, split_seed, (0.8, 0.1, 0.1))
    (val_filtered, val_num_rows_deleted) = drop_unknown(val, train)
    (test_filtered, test_num_rows_deleted) = drop_unknown(test, train)
    index_maps = build_index_maps(train)
    return PreparedData(
        train=EncodedRatings(*encode_ratings(train, index_maps)),
        val=EncodedRatings(*encode_ratings(val_filtered, index_maps)),
        test=EncodedRatings(*encode_ratings(test_filtered, index_maps)),
        index_maps=index_maps,
        val_num_rows_deleted=val_num_rows_deleted,
        test_num_rows_deleted=test_num_rows_deleted,
    )


def compute_residuals(data: PreparedData, reg_movie, reg_user) -> BiasBaselineResiduals:
    n_users = len(data.index_maps.user_ids)
    n_movies = len(data.index_maps.movie_ids)
    bias_model = BiasBaseline(
        n_users=n_users, n_movies=n_movies, reg_user=reg_user, reg_movie=reg_movie
    )
    bias_model.fit(data.train.users, data.train.movies, data.train.ratings)
    train_residuals = data.train.ratings - bias_model.predict(
        data.train.users, data.train.movies
    )
    val_residuals = data.val.ratings - bias_model.predict(
        data.val.users, data.val.movies
    )
    test_residuals = data.test.ratings - bias_model.predict(
        data.test.users, data.test.movies
    )

    return BiasBaselineResiduals(
        train_residuals=train_residuals,
        val_residuals=val_residuals,
        test_residuals=test_residuals,
        bias_model=bias_model,
    )
