from typing import Tuple

import numpy as np
import pandas as pd

def fix_random_seed(seed: int) -> None:
    """Fix the random seed for reproducibility
    seed: Seed for the random number generator
    """
    np.random.seed(seed)   

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean squared error between the true and predicted values
    y_true: True values
    y_pred: Predicted values
    
    Returns:
    mse: Mean squared error
    """
    mse = np.mean((y_true - y_pred)**2)
    return mse

def train_test_split(data: pd.DataFrame, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split the given data into a training and test set
    data: Data to split
    test_size: Fraction of data to use for the test set
    random_state: Seed for the random number generator
    
    Returns:
    train: Training set
    test: Test set
    """
    fix_random_seed(random_state)
    test = data.sample(frac=test_size, random_state=random_state)
    train = data.drop(test.index)
    return train, test
