"""
Logistic Regression — Built From Scratch
==========================================
Binary classification using sigmoid activation + BCE loss.
No sklearn for the core algorithm — pure NumPy.

Author: Your Name
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class LogisticRegressionScratch:
    """
    Binary Logistic Regression via Gradient Descent.

    Parameters
    ----------
    learning_rate : float  — gradient step size
    epochs        : int    — training iterations
    threshold     : float  — decision boundary (default 0.5)
    """

    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000, threshold: float = 0.5):
        self.lr = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Squash any real value to (0, 1) — the probability of class 1."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _bce_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary Cross-Entropy loss — penalises confident wrong predictions heavily."""
        eps = 1e-9  # numerical stability
        return float(-np.mean(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        ))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Gradient descent training loop."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Forward: linear combo → sigmoid probability
            z = X @ self.weights + self.bias
            y_prob = self._sigmoid(z)

            # Gradients of BCE w.r.t. w and b
            error = y_prob - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0:
                self.loss_history.append(self._bce_loss(y, y_prob))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class-1 probabilities for each sample."""
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary class labels using the decision threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=150)
    plt.show()


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic binary classification dataset
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=6,
        n_redundant=2, random_state=42
    )

    # Normalise features (crucial for gradient descent stability)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = LogisticRegressionScratch(learning_rate=0.1, epochs=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n✅ Logistic Regression (Scratch)")
    print(f"   Accuracy : {model.accuracy(X_test, y_test):.4f}")
    print("\n" + classification_report(y_test, y_pred))

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(0, 1000, 100), model.loss_history, color="#8e44ad", linewidth=2, marker="o")
    plt.title("Binary Cross-Entropy Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("assets/logistic_loss.png", dpi=150)
    plt.show()

    plot_confusion_matrix(y_test, y_pred)
