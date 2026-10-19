import numpy as np

rng = np.random.default_rng(42)
order = rng.permutation(10)

training_data = order[0:8]
validation_data = order[8:9]
test_data = order[9:]

initial_array = np.sort(np.concatenate([training_data, validation_data, test_data]))

print(f"Training data: {training_data}")
print(f"Validation data: {validation_data}")
print(f"Test data: {test_data}")

print(f"Recovered initial array: {initial_array}")

assert np.array_equal(np.arange(10), initial_array)
