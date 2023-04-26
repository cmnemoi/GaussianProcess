from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt # Graphiques (bas niveau)
import seaborn as sns # Graphiques (haut niveau)
sns.set()

def plot_train_and_val(
        X_train: np.ndarray, 
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        y_val_pred: np.ndarray
) -> None:
    """Plot the training and validation data and the predictions for both sets.
    X_train: Training data
    y_train: Training labels
    y_train_pred: Predictions for the training data
    X_val: Validation data
    y_val: Validation labels
    y_val_pred: Predictions for the validation data
    """
    plt.figure(figsize=(5, 10))
    fig, axes = plt.subplots(1, 2)
    sns.scatterplot(x=X_train, y=y_train, ax=axes[0])
    sns.lineplot(x=X_train, y=y_train_pred, color='red', ax=axes[0])
    sns.scatterplot(x=X_val, y=y_val, ax=axes[1])
    sns.lineplot(x=X_val, y=y_val_pred, color='red', ax=axes[1])

    fig.suptitle("Logarithme du revenu en fonction de l'âge")
    axes[0].set_xlabel("Âge en années")
    axes[0].set_ylabel("Logarithme du revenu")
    axes[1].set_xlabel("Âge en années")
    axes[1].set_ylabel("Logarithme du revenu")
    axes[0].legend(['Données d\'entraînement', 'Prédiction'])
    axes[1].legend(['Données de validation', 'Prédiction'], loc='lower right')

def plot_confidence_and_interval(gp, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Plot the prediction and the 95% confidence interval for the given Gaussian process
    gp: Gaussian process
    X: Input values
    y: Output values

    Returns:
    lower_bound: Lower bound of the confidence interval
    upper_bound: Upper bound of the confidence interval
    """
    mean_prediction, std_prediction = gp.predict(X)
    lower_bound = mean_prediction - 1.96 * np.sqrt(np.diag(std_prediction))
    upper_bound = mean_prediction + 1.96 * np.sqrt(np.diag(std_prediction))
    unique = np.unique(X, return_index=True)[1] # avoid a bug with plt.fill_between

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X, y=y, color="tab:blue", label="Données")
    sns.lineplot(x=X, y=mean_prediction, color="tab:red", label="Prédiction")
    plt.fill_between(
        X.ravel()[unique],
        lower_bound.ravel()[unique],
        upper_bound.ravel()[unique],
        color="tab:orange",
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.xlabel("Âge en années")
    plt.ylabel("Logarithme du revenu")
    plt.legend()
    plt.title("Prédiction et intervalle de confiance à 95%")

    return lower_bound, upper_bound
    
