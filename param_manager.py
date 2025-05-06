import json
import os

PARAM_FILE = "gp_params.json"

def load_params(func_id):
    """
    Load parameter configuration for a specific function ID (F1 to F8).
    Returns a dictionary of parameters.
    """
    key = f"F{func_id}"
    if not os.path.exists(PARAM_FILE):
        raise FileNotFoundError(f"Missing {PARAM_FILE}.")
    with open(PARAM_FILE, "r") as f:
        data = json.load(f)
    return data.get(key, {})

def update_params(func_id, new_values):
    """
    Update parameter configuration for a function ID in the parameter file.
    """
    key = f"F{func_id}"
    if not os.path.exists(PARAM_FILE):
        data = {}
    else:
        with open(PARAM_FILE, "r") as f:
            data = json.load(f)
    data[key] = new_values
    with open(PARAM_FILE, "w") as f:
        json.dump(data, f, indent=2)

def summarize_params():
    """
    Return all function parameters as a dictionary.
    """
    if not os.path.exists(PARAM_FILE):
        return {}
    with open(PARAM_FILE, "r") as f:
        return json.load(f)
