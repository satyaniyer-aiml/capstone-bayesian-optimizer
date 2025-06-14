import os
import json
from datetime import datetime

def extract_last_line(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if not lines:
            return ""
        return lines[-1].strip()

def extract_last_values(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if not lines:
            return ""
        last_line = lines[-1].strip()
        values = last_line.split(',')
        return values

def extract_gp_params_values(filepath, func_key):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
        gp_params = data.get(func_key, {})
        return [
            gp_params.get("acquisition", ""),
            gp_params.get("kernel", ""),
            str(gp_params.get("kappa", "")),
            str(gp_params.get("constant_value", "")),
            str(gp_params.get("length_scale", "")),
            str(gp_params.get("nu", ""))
        ]

def generate_journal_entry():
    today = datetime.now().strftime("%d-%b")
    gp_params_path = os.path.join("gp_params.json")
    output_lines_csv = []
    output_lines_tab = []

    for func_num in range(1, 9):
        func_key = f"F{func_num}"
        obs_path = os.path.join("data", f"observations_f{func_num}.csv")
        log_path = os.path.join("logs", f"log_func_{func_num}.csv")

        observation = extract_last_line(obs_path)
        log_values = extract_last_values(log_path)
        best_y_so_far = log_values[-1] if log_values else ""

        gp_params_values = extract_gp_params_values(gp_params_path, func_key)

        # Clean observation to remove trailing comma if it exists
        observation_values = observation.rstrip(',')
        
        # CSV-style output: observation values cleaned
        csv_line = f"{today}, {func_key}, {observation_values},{best_y_so_far}," + ",".join(gp_params_values)
        output_lines_csv.append(csv_line)

        # Tab-separated output: observation values stay comma-separated, then two tabs, then best_y_so_far, etc.
        observation_list = observation_values.split(',')
        tab_line = f"{today}, {func_key}, " + ",".join(observation_list) + "\t\t" + best_y_so_far + "\t" + "\t".join(gp_params_values)
        output_lines_tab.append(tab_line)

    return output_lines_csv, output_lines_tab

# Example usage
if __name__ == "__main__":
    csv_entries, tab_entries = generate_journal_entry()
    
    print("CSV-style output:")
    for entry in csv_entries:
        print(entry)
    
    print("\nTab-separated output:")
    for entry in tab_entries:
        print(entry)
