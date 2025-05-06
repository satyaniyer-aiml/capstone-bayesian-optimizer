"""
Smart Scheduler Main Script
- Loops over all 8 functions each week
- Chooses acquisition strategy, kernel, and kappa based on function characteristics
- Loads initial data from .npy if needed
- Logs each step for easy tracking
- Uses real black-box outputs only (no simulation)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from model import BlackBoxOptimizer
from utils import load_data, save_data, save_model, load_initial_data, log_progress, format_suggestion

# Ignore common warnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Function metadata ===
function_dims = {
    1: 2, 2: 2, 3: 3, 4: 4,
    5: 4, 6: 5, 7: 6, 8: 8
}

# === Function-specific strategy map ===
function_strategies = {
    1: {'acq': 'ucb', 'kernel': 'matern52', 'kappa': 2.5},   # sparse, multi-modal
    2: {'acq': 'ucb', 'kernel': 'matern52', 'kappa': 2.0},   # noisy, bumpy
    3: {'acq': 'ei',  'kernel': 'matern52', 'kappa': 1.96},  # maybe irrelevant dim
    4: {'acq': 'ei',  'kernel': 'matern52', 'kappa': 1.96},  # many local optima
    5: {'acq': 'ei',  'kernel': 'matern52', 'kappa': 1.96},  # unimodal, smooth
    6: {'acq': 'ucb', 'kernel': 'matern52', 'kappa': 2.0},   # mixed objectives
    7: {'acq': 'ei',  'kernel': 'matern52', 'kappa': 1.96},  # ML hyperparam
    8: {'acq': 'ei',  'kernel': 'rbf',      'kappa': 1.96}   # high-dimensional
}

# === Optimization loop ===
for func_id in range(1, 9):
    print(f"\n=== Function ID {func_id} ===")
    dim = function_dims[func_id]
    bounds = [(0, 1)] * dim

    # Load saved or initial data
    X_train, y_train = load_data(func_id)
    if X_train is None or y_train is None or len(X_train) == 0:
        X_train, y_train = load_initial_data(func_id)
        if X_train is not None and y_train is not None:
            print(f"? Loaded initial data for Function {func_id} from initial_data/")
        else:
            print(f"?? No initial data found for Function {func_id}. Starting empty.")
            X_train = np.empty((0, dim))
            y_train = np.array([])

    num_points = X_train.shape[0]

    # Display all observations so far
    if num_points > 0:
        print("?? All processed observations so far:")
        for i in range(num_points):
            x_formatted = format_suggestion(X_train[i].reshape(1, -1))
            print(f"  {i+1:02d}: {x_formatted} ? {y_train[i]:.6f}" if not np.isnan(y_train[i]) else f"  {i+1:02d}: {x_formatted} ? MISSING")

    # Get function-specific strategy
    strategy = function_strategies[func_id]
    acquisition_type = strategy['acq']
    kernel_type = strategy['kernel']
    kappa = strategy['kappa']

    print(f"Using Acquisition: {acquisition_type.upper()}, Kernel: {kernel_type.upper()}, Kappa: {kappa}")

    # Initialize model
    model = BlackBoxOptimizer(
        X_train, y_train,
        acquisition_type=acquisition_type,
        kernel_type=kernel_type,
        kappa=kappa,
        verbose=True
    )

    # Suggest input
    x_next = model.suggest_next(bounds)
    formatted_input = format_suggestion(x_next)
    print("?? Suggested next input (submit this):", formatted_input)

    # Enter observed output manually or skip
    raw_input_val = input(f"Enter observed output for Function {func_id} (or leave blank to store without output): ").strip()
    if raw_input_val == "":
        y_next = np.nan
        print(f"?? Recorded Function {func_id} input with missing output.")
    else:
        try:
            y_next = float(raw_input_val)
        except ValueError:
            print(f"? Invalid input for Function {func_id}. Skipping.")
            continue

    # Update training data and save
    X_train = np.vstack((X_train, x_next))
    y_train = np.append(y_train, y_next)

    save_data(X_train, y_train, func_id)
    save_model(model, func_id)

    # Log progress if output exists
    if not np.isnan(y_next):
        best_y_so_far = np.nanmax(y_train)
        log_progress(func_id, num_points + 1, acquisition_type, kernel_type, best_y_so_far)
        print(f"? Model for Function {func_id} updated and logged.")
    else:
        print(f"?? Output missing — model not updated or logged yet.")

print("\n?? All functions processed. Ready for next round.")
