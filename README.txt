# Capstone Project – Bayesian Optimization of Black-Box Functions

This project is a capstone exercise focused on using **Bayesian Optimization with Gaussian Processes (GPs)** to maximize 8 unknown black-box functions under evaluation constraints. The project includes a smart, adaptive scheduler that recommends the next input along with acquisition strategies and kernel parameters.

## 🔍 Project Features

- Supports multiple acquisition strategies: UCB, EI, PI
- Dynamically adjusts GP hyperparameters (length scale, constant, kernel type)
- Tracks and evolves model parameters across iterations via `gp_params.json`
- Saves and resumes state across restarts (data, models, logs)
- Modular design for plotting, parameter management, and model training

## 📁 Project Structure

```
.
├── main.py                 # Core script to run weekly optimization loop
├── model.py                # Gaussian Process optimizer with adaptive kernel
├── utils.py                # Data I/O, plotting, and helper functions
├── param_manager.py        # Handles loading/saving GP parameters
├── gp_params.json          # Stores function-specific GP settings
├── initial_data/           # Initial inputs/outputs (.npy format)
├── data/                   # Stores observations per function (CSV)
├── models/                 # Pickled GP models
├── logs/                   # Optimization logs per function
├── plot_gp_individual.py   # Optional GP visualization for 1D/2D functions
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 How to Run

```bash
python main.py
```

You will be prompted to submit for all 8 functions, one at a time. The system:
- Suggests the next input
- Logs progress
- Stores updated model and recommendation

## 🧠 Parameter Tracking

Hyperparameters for each function (e.g., kernel type, kappa, length scale) are saved to `gp_params.json` and updated automatically after each model training.

## 📊 Visualizing

You can visualize any 1D or 2D GP model with:

```bash
python plot_gp_individual.py
```

## 🔧 Dependencies

Install packages with:

```bash
pip install -r requirements.txt
```

## 📜 License

This project is open for academic and personal use.
