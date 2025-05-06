"""
Defines 8 synthetic black-box functions of increasing complexity.
Used for development/testing without needing real function evaluations.
"""

import numpy as np

def evaluate(func_id, x):
    """
    Evaluate one of 8 predefined black-box functions.
    Args:
        func_id: int (1-8)
        x: np.array of shape (1, d)
    Returns:
        float: scalar output of black-box function
    """
    x = x.ravel()
    if func_id == 1:
        return np.sin(2 * np.pi * x[0])
    elif func_id == 2:
        return np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])
    elif func_id == 3:
        return np.sum(np.sin(5 * np.pi * x))
    elif func_id == 4:
        return np.exp(-np.sum((x - 0.5)**2) * 10)
    elif func_id == 5:
        return np.prod(np.cos(x * np.pi)) + np.sum(x**2)
    elif func_id == 6:
        return np.sum(x**2 - np.cos(2 * np.pi * x))
    elif func_id == 7:
        return -np.sum(np.sin(x) * np.sin((x**2) / np.pi))
    elif func_id == 8:
        return np.sum((x - 0.3)**2) + 0.1 * np.sin(10 * x[0])
    else:
        raise ValueError("func_id must be between 1 and 8")
