from dataclasses import dataclass

import numpy as np

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
