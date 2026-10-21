import numpy as np

from spectralrec.metrics import mae, rmse

y_true = np.array([[1], [2], [3]])
y_pred = np.array([[3], [4], [5]])

print(f"MAE {y_true.tolist()}, {y_pred.tolist()}: {mae(y_true, y_pred):.2f}")


y_true = y_true.reshape(1, -1)
y_pred = y_pred.reshape(1, -1)
print(f"RMSE {y_true.tolist()}, {y_pred.tolist()}: {rmse(y_true, y_pred):.2f}")
