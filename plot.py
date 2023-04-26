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
