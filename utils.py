"""
Utility functions for data I/O, GP plotting, and progress logging.
Assumes real black-box mode (no simulation).
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

def load_data(func_id, base="data"):
    """Load saved CSV data for a given function ID."""
    path = f"{base}/observations_f{func_id}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values
    return None, None

def save_data(X, y, func_id, base="data"):
    """Save input-output observations for a function."""
    os.makedirs(base, exist_ok=True)
    df = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))))
    df.to_csv(f"{base}/observations_f{func_id}.csv", index=False)

def save_model(model, func_id, base="models"):
    """Save a trained GP model using joblib."""
    os.makedirs(base, exist_ok=True)
    joblib.dump(model, f"{base}/gp_model_f{func_id}.pkl")

def load_model(func_id, base="models"):
    """Load a previously saved GP model."""
    path = f"{base}/gp_model_f{func_id}.pkl"
    return joblib.load(path) if os.path.exists(path) else None

def load_initial_data(func_id, base="initial_data"):
    """Load initial .npy data from provided folders."""
    folder = os.path.join(base, f"function_{func_id}")
    x_path = os.path.join(folder, "initial_inputs.npy")
    y_path = os.path.join(folder, "initial_outputs.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        return np.load(x_path), np.load(y_path)
    return None, None

def plot_gp(model, bounds, dim, X_train, y_train):
    """Plot GP mean prediction and uncertainty for 1D or 2D cases."""
    if dim == 1:
        x = np.linspace(bounds[0][0], bounds[0][1], 500).reshape(-1, 1)
        mu, sigma = model.predict(x)
        plt.figure()
        plt.plot(x, mu, 'b-', label='Mean Prediction')
        plt.fill_between(x.ravel(), mu - 1.96*sigma, mu + 1.96*sigma, alpha=0.2, label='Confidence Interval')
        plt.plot(X_train, y_train, 'ro', label='Observations')
        plt.title("GP Prediction (1D)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

        # Fit quality check
        mu_train, _ = model.predict(X_train)
        mae = np.mean(np.abs(mu_train - y_train))
        if mae < 0.1:
            print("? GP fitting looks good: model matches training points.")
        else:
            print("?? Warning: GP fitting may be poor; mean deviates from data.")

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
        plt.title("GP Mean Prediction (2D)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid()
        plt.show()

def log_progress(func_id, iteration, acquisition_type, kernel_type, best_y, base="logs"):
    """Append a log entry per iteration for progress tracking."""
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"log_func_{func_id}.csv")
    entry = {
        'iteration': iteration,
        'acquisition': acquisition_type,
        'kernel': kernel_type,
        'best_y_so_far': best_y
    }
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    df.to_csv(path, index=False)

def format_suggestion(x):
    """
    Format a numpy array x into the required capstone string format:
    - 6 decimal places
    - hyphen-separated
    - 0.0 becomes 0.000000
    - Values are clipped just below 1.0 (max is 0.999999)
    """
    clipped = np.clip(x.ravel(), 0, 0.999999)
    return '-'.join(f"{val:.6f}" for val in clipped)
