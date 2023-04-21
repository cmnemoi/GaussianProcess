from typing import Tuple

from numpy.linalg import inv
from scipy.optimize import minimize
import numpy as np

class GaussianProcess:
    def __init__(self, kernel: str = 'matern') -> None:
        """Gaussian process."""
        self.theta = 1e-1
        self.noise = 1e-5
        self.kernel = kernel
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Gaussian process to the given data.
        X: List of input values
        y: List of output values
        """
        self.X = X
        self.y = y
        self.kii = self._get_covariance_matrix(X, X)
        self.kii += self.noise * np.eye(len(X))

    def optimize(self, X, y) -> None:
        """Optimize the kernel parameter theta and the noise parameter noise using the given data
        X: List of input values
        y: List of output values
        """

        def negative_log_likelihood(hyper_parameters: np.ndarray) -> float:
            self.theta, self.noise = hyper_parameters
            return -self._log_likelihood(X, y)[0]
        
        res = minimize(negative_log_likelihood, (self.theta, self.noise), method='L-BFGS-B', bounds=((1e-5, None), (1e-5, None)))
        self.theta = res.x[0]
        self.noise = res.x[1]

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the mean prediction and its covariance matrix for the given input X
        X: List of input values

        Returns:
        zeta: Mean prediction
        sigma: Covariance matrix
        """
        prediction_matrix = X
        training_matrix = self.X

        kx = self._get_covariance_matrix(prediction_matrix, training_matrix)
        zeta = kx @ inv(self.kii + self.noise**2 * np.eye(len(self.X))) @ self.y
        sigma = self._get_covariance_matrix(prediction_matrix, prediction_matrix) - kx @ inv(self.kii + self.noise**2 * np.eye(len(self.X))) @ kx.T
        
        return zeta.ravel(), sigma
    
    def sample(self, X, n=1) -> np.ndarray:
        """Return n samples from the Gaussian process for the given input X
        X: List of input values
        y: List of output values
        n: Number of processes to sample
        
        Returns:
        samples: Samples from the Gaussian process
        """
        zeta, sigma = self.predict(X)
        return np.random.multivariate_normal(zeta, sigma, n)
    
    def score(self, X, y) -> float:
        """Return the mean squared error of prediction for the given (X, y) data
        X: List of input values
        y: List of output values
        
        Returns:
        mse: Mean squared error
        """
        return np.mean((self.predict(X)[0] - y)**2)

    def _get_covariance_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Return the covariance matrix between the given matrixes
        A, B: Matrixes

        Returns:
        covariance_matrix: Covariance matrix
        """
        return self._kernel(A[:, np.newaxis], B[np.newaxis, :])
    
    def _kernel(self, x1, x2) -> float:
        """Return the kernel value between the given input values x1 and x2"""
        match self.kernel:
            case 'exp_sin_squared':
                return self._exp_sin_squared_kernel(x1, x2)
            case 'matern':
                return self._matern_kernel(x1, x2)
            case 'rational_quadratic':
                return self._rational_quadratic_kernel(x1, x2)
            case _:
                raise ValueError(f'Kernel {self.kernel} not supported')
    
    def _log_likelihood(self, X, y) -> float:
        """Return the negative log-likelihood of the given data
        X: List of input values
        y: List of output values
        
        Returns:
        log_likelihood: Negative log-likelihood
        """
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        self.fit(X, y)
        return -0.5 * y.T @ inv(self.kii) @ y - 0.5 * np.log(np.linalg.det(self.kii) + 1e-8) - 0.5 * len(X) * np.log(2*np.pi)

    def _exp_sin_squared_kernel(self, x1, x2) -> float:
        """Return the exponential sine squared kernel value between the given input values x1 and x2"""
        theta = self.theta + 1e-8
        return np.exp(-2 * np.sin(np.pi * np.abs(x1 - x2))**2 / theta**2)
    
    def _matern_kernel(self, x1, x2) -> float:
        """Return the Matern kernel value with mu = 5/2 between the given input values x1 and x2"""
        theta = self.theta + 1e-8
        return (1 + np.sqrt(5) * (np.abs(x1 - x2)) / theta + (5 * (x1 - x2)**2) / (3 * theta**2)) * np.exp(-np.sqrt(5) * (np.abs(x1 - x2)) / theta)
    
    def _rational_quadratic_kernel(self, x1, x2) -> float:
        """Return the rational quadratic kernel value between the given input values x1 and x2"""
        theta = self.theta + 1e-8
        return 1 + (x1 - x2)**2 / (2 * theta**2)
    
    
        