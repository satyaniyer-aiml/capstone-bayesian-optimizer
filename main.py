"""
Smart Scheduler Main Script
- Loops over all 8 functions each week
- Loads initial data from .npy if needed
- Uses parameters from gp_params.json
- Logs each step for easy tracking
- Outputs next suggestions to next_suggestions.txt
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
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

# === Initialize suggestion file ===
suggestions_path = "next_suggestions.txt"
with open(suggestions_path, "w") as f:
    f.write("Next Suggestions (one per function):\n")

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
            print(f"Loaded initial data for Function {func_id} from initial_data/")
        else:
            print(f"No initial data found for Function {func_id}. Starting empty.")
            X_train = np.empty((0, dim))
            y_train = np.array([])

    num_points = X_train.shape[0]

    # Display all observations so far
    if num_points > 0:
        print("All processed observations so far:")
        for i in range(num_points):
            x_formatted = format_suggestion(X_train[i].reshape(1, -1))
            print(f"  {i+1:02d}: {x_formatted} ? {y_train[i]:.6f}" if not np.isnan(y_train[i]) else f"  {i+1:02d}: {x_formatted} ? MISSING")

    # Initialize model
    model = BlackBoxOptimizer(X_train, y_train, func_id, verbose=True)

    print(f"Using Acquisition: {model.acquisition_type.upper()}, Kernel: {model.kernel_type.upper()}, Kappa: {model.kappa}")

    # Suggest input
    x_next = model.suggest_next(bounds)
    formatted_input = format_suggestion(x_next)
    print("Suggested next input (submit this):", formatted_input)
    with open(suggestions_path, "a") as f:
        f.write(f"Function {func_id}: {formatted_input}\n")

    # Enter observed output manually or skip
    raw_input_val = input(f"Enter observed output for Function {func_id} (or leave blank to store without output): ").strip()
    if raw_input_val == "":
        y_next = np.nan
        print(f"Recorded Function {func_id} input with missing output.")
    else:
        try:
            y_next = float(raw_input_val)
        except ValueError:
            print(f"Invalid input for Function {func_id}. Skipping.")
            continue

    # Format x_next to ensure consistency in submission and storage
    formatted_values = np.array([[float(val) for val in formatted_input.split('-')]])
    X_train = np.vstack((X_train, formatted_values))
    y_train = np.append(y_train, y_next)

    save_data(X_train, y_train, func_id)
    save_model(model, func_id)

    # Log progress if output exists
    best_y_so_far = np.nanmax(y_train)
    log_progress(func_id, num_points + 1, model.acquisition_type, model.kernel_type, best_y_so_far)
    print(f"Model for Function {func_id} updated and logged.")

print("\nAll functions processed. Ready for next round.")
