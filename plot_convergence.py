"""
Plot convergence graphs for each function (Function ID 1 to 8)
Shows how the best observed output improves over iterations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# === Settings ===
logs_folder = "logs"
output_folder = "plots"

os.makedirs(output_folder, exist_ok=True)

# === Plot convergence for each function ===
for func_id in range(1, 9):
    log_path = f"{logs_folder}/log_func_{func_id}.csv"

    if not os.path.exists(log_path):
        print(f"Log not found for Function {func_id}. Skipping...")
        continue

    df = pd.read_csv(log_path)
    iterations = df['iteration']
    best_y = df['best_y_so_far']

    plt.figure()
    plt.plot(iterations, best_y, marker='o')
    plt.title(f"Convergence Plot - Function {func_id}")
    plt.xlabel("Iteration (Week)")
    plt.ylabel("Best Observed Output")
    plt.grid()
    plt.savefig(f"{output_folder}/convergence_func_{func_id}.png")
    plt.close()

    print(f"Saved convergence plot for Function {func_id}.")

print("\n? All convergence plots saved in 'plots/' folder.")
