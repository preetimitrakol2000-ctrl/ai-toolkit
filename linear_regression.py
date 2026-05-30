"""
Linear Regression — Built From Scratch
=======================================
No sklearn. Pure NumPy gradient descent.
Demonstrates: MSE loss, weight updates, convergence.

Author: Your Name
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionScratch:
    """
    Ordinary Least Squares Linear Regression using Gradient Descent.

    Parameters
    ----------
    learning_rate : float
        Step size for each gradient update. Default 0.01.
    epochs : int
        Number of full passes over the training data. Default 1000.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def _mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error loss."""
        return float(np.mean((y_true - y_pred) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        """
        Train the model using gradient descent.

        Parameters
        ----------
        X : shape (n_samples, n_features)
        y : shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Gradients (partial derivatives of MSE)
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # Parameter update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Track loss every 50 epochs
            if epoch % 50 == 0:
                loss = self._mse(y, y_pred)
                self.loss_history.append(loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values for input X."""
        if self.weights is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score — how much variance the model explains."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def plot_loss(self) -> None:
        """Plot training loss curve over epochs."""
        plt.figure(figsize=(8, 4))
        plt.plot(range(0, self.epochs, 50), self.loss_history, color="#e74c3c", linewidth=2)
        plt.title("Training Loss (MSE) Over Epochs", fontsize=14)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("assets/loss_curve.png", dpi=150)
        plt.show()
        print("Loss curve saved → assets/loss_curve.png")


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # Synthetic dataset: y = 3x + 5 + noise
    X = np.random.randn(200, 1)
    y = 3 * X.squeeze() + 5 + np.random.randn(200) * 0.8

    # Train / test split (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train
    model = LinearRegressionScratch(learning_rate=0.05, epochs=500)
    model.fit(X_train, y_train)

    # Evaluate
    r2 = model.score(X_test, y_test)
    print(f"\n✅ Linear Regression (Scratch)")
    print(f"   Learned weight : {model.weights[0]:.4f}  (true: 3.0)")
    print(f"   Learned bias   : {model.bias:.4f}       (true: 5.0)")
    print(f"   R² on test set : {r2:.4f}")

    # Visualise
    model.plot_loss()

    plt.figure(figsize=(7, 5))
    plt.scatter(X_test, y_test, alpha=0.6, label="Actual", color="#3498db")
    plt.plot(X_test, model.predict(X_test), color="#e74c3c", linewidth=2, label="Predicted")
    plt.title("Linear Regression — Scratch vs Data", fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("assets/linear_regression_fit.png", dpi=150)
    plt.show()
