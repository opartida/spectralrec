from pathlib import Path

from spectralrec.data import drop_unknown, load_ratings, split_ratings

RATINGS_PATH = Path("data/raw/ml-latest-small/ratings.csv")
SPLIT_SEED = 42
ratings = load_ratings(RATINGS_PATH)


(train, val, test) = split_ratings(ratings, SPLIT_SEED, (0.8, 0.1, 0.1))

(_, num_rows_deleted_from_val) = drop_unknown(val, train)


(_, num_rows_deleted_from_test) = drop_unknown(test, train)

print(f"Num rows deleted from validation: {num_rows_deleted_from_val}")
print(f"Num rows deleted from test: {num_rows_deleted_from_test}")

print(
    f"Percentage of rows deleted from validation: {num_rows_deleted_from_val / val.shape[0]:.2%}"
)
print(
    f"Percentage of rows deleted from test: {num_rows_deleted_from_test / test.shape[0]:.2%}"
)
