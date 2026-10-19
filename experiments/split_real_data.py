from pathlib import Path

from spectralrec.data import drop_unknown, load_ratings, split_ratings

RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")
SPLIT_SEED = 42
ratings = load_ratings(RATINGS_PATH)


(training, validation, testing) = split_ratings(ratings, SPLIT_SEED, (0.8, 0.1, 0.1))

(_, num_rows_deleted_from_validation) = drop_unknown(validation, training)


(_, num_rows_deleted_from_testing) = drop_unknown(testing, training)

print(f"Num rows deleted from validation: {num_rows_deleted_from_validation}")
print(f"Num rows deleted from testing: {num_rows_deleted_from_testing}")

print(
    f"Percentage of rows deleted from validation: {num_rows_deleted_from_validation / validation.shape[0]:.2%}"
)
print(
    f"Percentage of rows deleted from testing: {num_rows_deleted_from_testing / testing.shape[0]:.2%}"
)
