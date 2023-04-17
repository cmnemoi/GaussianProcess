import numpy as np   

class ExponentialTransformer():
    def __init__(self):
        self.min = None

    def fit(self, X: np.ndarray) -> None:
        """Fit the transformation to the given data.
        X: List of input values
        """
        self.min = np.min(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the given data.
        X: List of input values
        
        Returns:
        X: Transformed data
        """
        X -= self.min
        X = np.exp(X)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform the given data.
        X: List of input values
        
        Returns:
        X: Inverse transformed data
        """
        X = np.log(X)
        X += self.min
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the transformation to the given data and transform it.
        X: List of input values
        
        Returns:
        X: Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
class StandardScaler():
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> None:
        """Fit the scaler to the given data.
        X: List of input values
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the given data.
        X: List of input values
        
        Returns:
        X: Transformed data
        """
        X -= self.mean
        X /= self.std
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform the given data.
        X: List of input values
        
        Returns:
        X: Inverse transformed data
        """
        X *= self.std
        X += self.mean
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the scaler to the given data and transform it.
        X: List of input values
        
        Returns:
        X: Transformed data
        """
        self.fit(X)
        return self.transform(X)