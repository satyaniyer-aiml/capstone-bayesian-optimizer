import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from scipy.optimize import minimize
from scipy.stats import norm
from param_manager import load_params, update_params

class BlackBoxOptimizer:
    def __init__(self, X_train, y_train, func_id, verbose=False):
        """
        Initialize the Gaussian Process optimizer using parameters from gp_params.json.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.verbose = verbose

        # Load config
        self.params = load_params(func_id)
        self.acquisition_type = self.params.get("acquisition", "ucb")
        self.kernel_type = self.params.get("kernel", "matern52")
        self.kappa = self.params.get("kappa", 1.96)

        # Build kernel from parameter config
        const_val = self.params.get("constant_value", 1.0)
        length_scale = self.params.get("length_scale", 1.0)

        if self.kernel_type == 'matern52':
            nu = 2.5
        elif self.kernel_type == 'matern32':
            nu = 1.5
        else:
            nu = self.params.get("nu", 2.5)  # fallback

        if self.kernel_type in ['matern52', 'matern32']:
            kernel = ConstantKernel(const_val) * Matern(length_scale=length_scale, nu=nu) + WhiteKernel()
        elif self.kernel_type == 'rbf':
            kernel = ConstantKernel(const_val) * RBF(length_scale=length_scale) + WhiteKernel()
        else:
            raise ValueError("Unsupported kernel type")

        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.update_model()

    def update_model(self):
        if len(self.X_train) > 0:
            self.gp.fit(self.X_train, self.y_train)

            # Save learned parameters back to config
            k = self.gp.kernel_
            if self.kernel_type in ['matern52', 'matern32']:
                learned = {
                    "acquisition": self.acquisition_type,
                    "kernel": self.kernel_type,
                    "kappa": self.kappa,
                    "constant_value": k.k1.k1.constant_value,
                    "length_scale": k.k1.k2.length_scale,
                    "nu": k.k1.k2.nu
                }
            elif self.kernel_type == 'rbf':
                learned = {
                    "acquisition": self.acquisition_type,
                    "kernel": self.kernel_type,
                    "kappa": self.kappa,
                    "constant_value": k.k1.k1.constant_value,
                    "length_scale": k.k1.k2.length_scale,
                    "nu": None
                }
            update_params(self.params.get("function_id", 0), learned)

    def predict(self, X):
        return self.gp.predict(X, return_std=True)

    def acquisition(self, X):
        mu, sigma = self.predict(X)
        if self.acquisition_type == 'ucb':
            return mu + self.kappa * sigma
        elif self.acquisition_type == 'ei':
            y_best = np.max(self.y_train)
            z = (mu - y_best) / (sigma + 1e-9)
            return (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
        elif self.acquisition_type == 'poi':
            y_best = np.max(self.y_train)
            z = (mu - y_best) / (sigma + 1e-9)
            return norm.cdf(z)
        else:
            raise ValueError("Unsupported acquisition type")

    def suggest_next(self, bounds, n_restarts=10):
        if self.X_train.shape[0] == 0:
            return np.array([[np.random.uniform(b[0], b[1]) for b in bounds]])

        best_x = None
        best_acq = -np.inf
        for _ in range(n_restarts):
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            res = minimize(lambda x: -self.acquisition(x.reshape(1, -1)), x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.success and -res.fun > best_acq:
                best_acq = -res.fun
                best_x = res.x

        if best_x is None:
            best_x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])

        if self.verbose:
            print(f"Best acquisition value: {best_acq:.4f}")

        return best_x.reshape(1, -1)
