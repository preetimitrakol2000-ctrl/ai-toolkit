"""
K-Nearest Neighbors Classifier — Built From Scratch
=====================================================
No sklearn for the core algorithm. Uses Euclidean distance
and majority-vote to classify new data points.

Key insight: KNN is lazy — it memorises all training data
and only does computation at prediction time.

Author: Your Name
"""

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class KNNClassifier:
    """
    K-Nearest Neighbors (KNN) Classifier from scratch.

    Parameters
    ----------
    k : int — number of neighbors to consider (default 3)
    """

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """KNN 'training' is just storing the data — no computation yet."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _euclidean_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distances from one point to all training samples."""
        return np.sqrt(np.sum((self.X_train - point) ** 2, axis=1))

    def _predict_single(self, point: np.ndarray) -> int:
        """Find k nearest neighbors and return the majority class."""
        distances = self._euclidean_distances(point)
        k_indices = np.argsort(distances)[: self.k]        # indices of k smallest distances
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        return int(most_common)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for all samples in X."""
        return np.array([self._predict_single(x) for x in X])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


def find_best_k(X_train, y_train, X_val, y_val, k_range=range(1, 21)):
    """Sweep over k values and return accuracy at each."""
    accuracies = []
    for k in k_range:
        model = KNNClassifier(k=k)
        model.fit(X_train, y_train)
        accuracies.append(model.accuracy(X_val, y_val))
    return list(k_range), accuracies


def plot_decision_boundary(model, X, y, title="KNN Decision Boundary"):
    """2D decision boundary plot (uses first 2 features)."""
    X2 = X[:, :2]
    model_2d = KNNClassifier(k=model.k)
    model_2d.fit(X2, y)

    h = 0.05
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model_2d.predict(grid_points).reshape(xx.shape)

    colors = ["#AED6F1", "#A9DFBF", "#F9E79F"]
    dot_colors = ["#2980B9", "#27AE60", "#F39C12"]

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.get_cmap("Set1", 3))
    for cls, col in zip(np.unique(y), dot_colors):
        mask = y == cls
        plt.scatter(X2[mask, 0], X2[mask, 1], c=col, edgecolors="k",
                    linewidths=0.5, s=50, label=f"Class {cls}")
    plt.title(f"{title}  (k={model.k})", fontsize=13)
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/knn_boundary.png", dpi=150)
    plt.show()


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    # Scale features — KNN is distance-based, so scaling is crucial
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Find best k
    ks, accs = find_best_k(X_train, y_train, X_test, y_test)
    best_k = ks[np.argmax(accs)]

    print(f"\n✅ KNN Classifier (Scratch) on Iris Dataset")
    print(f"   Best k      : {best_k}")
    print(f"   Accuracy    : {max(accs):.4f}")

    # K vs Accuracy plot
    plt.figure(figsize=(8, 4))
    plt.plot(ks, accs, marker="o", color="#e67e22", linewidth=2)
    plt.axvline(best_k, linestyle="--", color="grey", alpha=0.7, label=f"Best k={best_k}")
    plt.title("KNN Accuracy vs. Number of Neighbors (k)")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("assets/knn_k_sweep.png", dpi=150)
    plt.show()

    # Decision boundary
    final_model = KNNClassifier(k=best_k)
    final_model.fit(X_train, y_train)
    plot_decision_boundary(final_model, X_scaled, y)
