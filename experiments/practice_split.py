import numpy as np

rng = np.random.default_rng(42)
order = rng.permutation(10)

train_data = order[0:8]
val_data = order[8:9]
test_data = order[9:]

initial_array = np.sort(np.concatenate([train_data, val_data, test_data]))

print(f"Train data: {train_data}")
print(f"Validation data: {val_data}")
print(f"Test data: {test_data}")

print(f"Recovered initial array: {initial_array}")

assert np.array_equal(np.arange(10), initial_array)
