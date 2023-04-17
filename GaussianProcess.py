from typing import List, Tuple

from scipy.optimize import minimize
import numpy as np

class GaussianProcess:
    def __init__(self, theta: float, noise: float):
        """Gaussian process with Matern kernel and zero mean function.
        theta: Kernel parameter
        noise: Noise parameter
        """
        self.theta = theta
        self.noise = noise
        self.mean = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Gaussian process to the given data.
        X: List of input values
        y: List of output values
        """
        self.mean = np.mean(y)
        self.X = X
        self.y = y
        kii = self._get_covariance_matrix(X, X)
        kii += self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(kii)

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
        zeta = kx @ np.linalg.cholesky(self.K_inv).T @ self.y
        sigma = self._get_covariance_matrix(prediction_matrix, prediction_matrix) - kx @ np.linalg.cholesky(self.K_inv).T @ kx.T
        
        return zeta, sigma
    
    def sample(self, X, y, n=1) -> np.ndarray:
        """Return n samples from the Gaussian process for the given input X
        X: List of input values
        y: List of output values
        n: Number of processes to sample
        
        Returns:
        samples: Samples from the Gaussian process
        """
        self.fit(X, y)
        zeta, sigma = self.predict(X)
        return np.random.multivariate_normal(zeta, sigma, n)
    
    def neg_log_likelihood(self, X, y) -> float:
        """Return the negative log-likelihood of the given data
        X: List of input values
        y: List of output values
        
        Returns:
        log_likelihood: Negative log-likelihood
        """
        self.fit(X, y)
        # TODO: justify this
        log_likelihood = -0.5 * y.T.dot(self.K_inv).dot(y) - 0.5 * np.log(np.linalg.det(self.K)) - 0.5 * len(X) * np.log(2*np.pi)
        return -log_likelihood
    
    def optimize(self, X, y) -> None:
        """Optimize the kernel parameter theta and the noise parameter noise using the given data
        X: List of input values
        y: List of output values
        """
        # TODO; code gradient descent from scratch?
        res = minimize(self.neg_log_likelihood, self.theta, args=(X, y), method='L-BFGS-B', bounds=((1e-5, None),))
        self.theta = res.x[0]

        res = minimize(self.neg_log_likelihood, self.noise, args=(X, y), method='L-BFGS-B', bounds=((1e-5, None),))
        self.noise = res.x[0]

    def _get_covariance_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Return the covariance matrix between the given matrixes
        A, B: Matrixes

        Returns:
        covariance_matrix: Covariance matrix
        """
        covariance_matrix = np.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                covariance_matrix[i,j] = self._kernel(A[i], B[j])
        return covariance_matrix
    
    def _kernel(self, x1, x2) -> float:
        """Matern kernel with nu=5/2"""
        return (1+np.sqrt(5)*(np.abs(x1-x2))/self.theta+(5*(x1-x2)**2)/(3*self.theta**2)) * np.exp(-np.sqrt(5)*(np.abs(x1-x2))/self.theta)
        