"""
Plot GP prediction (mean and uncertainty) for a selected function (1D or 2D only).
Run this manually to visualize model fit without blocking main workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_model

# === User input ===
func_id = int(input("Enter function ID to plot (1-8): "))

# === Dimension mapping (should match main script) ===
function_dims = {
    1: 2, 2: 2, 3: 3, 4: 4,
    5: 4, 6: 5, 7: 6, 8: 8
}

dim = function_dims[func_id]
bounds = [(0, 1)] * dim

# === Load data and model ===
X_train, y_train = load_data(func_id)
model = load_model(func_id)

if model is None or X_train is None or y_train is None or len(X_train) == 0:
    print("? No data or model found for this function.")
    exit()

# === Plot 1D ===
if dim == 1:
    x = np.linspace(bounds[0][0], bounds[0][1], 500).reshape(-1, 1)
    mu, sigma = model.predict(x)

    plt.figure()
    plt.plot(x, mu, 'b-', label='Mean Prediction')
    plt.fill_between(x.ravel(), mu - 1.96 * sigma, mu + 1.96 * sigma, alpha=0.2, label='Confidence Interval')
    plt.plot(X_train, y_train, 'ro', label='Observations')
    plt.title(f"Function {func_id} - GP Prediction (1D)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

# === Plot 2D ===
elif dim == 2:
    x1 = np.linspace(bounds[0][0], bounds[0][1], 50)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 50)
    X1, X2 = np.meshgrid(x1, x2)
    Xgrid = np.vstack([X1.ravel(), X2.ravel()]).T
    Z, _ = model.predict(Xgrid)
    Z = Z.reshape(X1.shape)

    plt.figure()
    contour = plt.contourf(X1, X2, Z, levels=30)
    plt.colorbar(contour)
    plt.plot(X_train[:, 0], X_train[:, 1], 'ro', label='Observations')
    plt.title(f"Function {func_id} - GP Mean Prediction (2D)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid()
    plt.show()

else:
    print(f"?? Plotting not supported for dimension {dim}. Only 1D or 2D functions.")
