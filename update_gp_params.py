import os
import json
import pandas as pd
import numpy as np

def detect_stagnation(log_path, window=5, threshold=1e-3):
    """
    Check if best_y_so_far has stopped improving.
    """
    if not os.path.exists(log_path):
        print(f"[INFO] Log file not found: {log_path}")
        return False, None

    df = pd.read_csv(log_path)
    if df.empty or df.shape[0] < window + 1:
        print(f"[INFO] Not enough data points in {log_path} to assess stagnation.")
        return False, None

    recent = df['best_y_so_far'].values[-(window + 1):]
    diffs = np.diff(recent)
    avg_improvement = np.mean(np.maximum(diffs, 0))
    print(f"[DEBUG] Avg improvement for {log_path}: {avg_improvement:.6f}")
    return avg_improvement < threshold, avg_improvement

def suggest_new_params(current_params, stagnated):
    """
    Suggests new parameters if stagnation is detected.
    """
    new_params = current_params.copy()
    reason = []

    if stagnated and current_params.get("acquisition") == "ucb":
        new_params["acquisition"] = "ei"
        reason.append("Switched acquisition from 'ucb' to 'ei' due to stagnation to encourage exploration.")
    elif stagnated and current_params.get("acquisition") == "ei":
        new_params["acquisition"] = "poi"
        reason.append("Switched acquisition from 'ei' to 'poi' due to continued stagnation.")

    if stagnated and current_params.get("kernel") == "matern52":
        new_params["kernel"] = "matern32"
        reason.append("Changed kernel from 'matern52' to 'matern32' to increase sensitivity to local changes.")

    if "kappa" in current_params:
        old_kappa = current_params["kappa"]
        new_kappa = min(old_kappa + 0.5, 10)
        if stagnated and new_kappa != old_kappa:
            new_params["kappa"] = new_kappa
            reason.append(f"Increased kappa from {old_kappa} to {new_kappa} to allow broader exploration.")

    if "constant_value" in current_params:
        old_val = current_params["constant_value"]
        new_val = round(old_val * 1.5, 2)
        new_params["constant_value"] = new_val
        reason.append(f"Increased constant_value from {old_val} to {new_val} to raise model's signal strength.")

    if "length_scale" in current_params:
        old_scale = current_params["length_scale"]
        new_scale = round(old_scale * 1.2, 2)
        new_params["length_scale"] = new_scale
        reason.append(f"Increased length_scale from {old_scale} to {new_scale} to smooth the GP's predictions.")

    if "nu" in current_params:
        old_nu = float(current_params.get("nu", 2.5))
        new_nu = min(old_nu + 0.5, 3.0)
        if stagnated and new_nu != old_nu:
            new_params["nu"] = new_nu
            reason.append(f"Increased nu from {old_nu} to {new_nu} to allow smoother kernel behavior.")

    return new_params, reason

def update_gp_params():
    """
    Main function to read logs, check for stagnation, and update gp_params.json.
    """
    gp_path = "gp_params.json"
    logs_dir = "logs"
    history_dir = "param_history"
    os.makedirs(history_dir, exist_ok=True)

    with open(gp_path, 'r') as f:
        gp_data = json.load(f)

    update_report = {}

    for key in gp_data:
        func_num = key[1:]  # 'F1' -> '1'
        log_file = os.path.join(logs_dir, f"log_func_{func_num}.csv")
        stagnated, improvement = detect_stagnation(log_file)

        print(f"[CHECK] {key}: stagnated = {stagnated}, avg_improvement = {improvement}")

        if stagnated:
            new_params, explanation = suggest_new_params(gp_data[key], stagnated)
            if new_params != gp_data[key]:
                update_report[key] = {
                    "old": gp_data[key],
                    "new": new_params,
                    "reason": explanation
                }
                gp_data[key] = new_params

                # Append new params to CSV history
                hist_path = os.path.join(history_dir, f"{key}_params.csv")
                new_row = {**new_params, "iteration_tag": pd.Timestamp.now().isoformat()}
                hist_df = pd.DataFrame([new_row])
                if os.path.exists(hist_path):
                    hist_df.to_csv(hist_path, mode='a', header=False, index=False)
                else:
                    hist_df.to_csv(hist_path, index=False)

    # Save updated params
    with open(gp_path, 'w') as f:
        json.dump(gp_data, f, indent=2)

    return update_report

# Execute update
if __name__ == "__main__":
    updates = update_gp_params()
    if not updates:
        print("No updates were made. All functions are performing adequately.")
    else:
        print("\n=== GP PARAMETER UPDATES ===")
        for func, details in updates.items():
            print(f"\n{func}:")
            for r in details["reason"]:
                print(f" - {r}")
            print(f"Old: {details['old']}")
            print(f"New: {details['new']}")
