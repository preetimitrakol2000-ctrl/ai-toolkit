"""
Visualizer — Plug-and-Play ML Plotting Utilities
=================================================
Drop-in functions for the most common ML plots:
  - Loss curves
  - Confusion matrices
  - Feature distributions
  - Correlation heatmaps
  - Learning curves (bias-variance)
  - Scatter matrix (pairplot)

Author: Your Name
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix


# ── Style config ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

PALETTE = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]


# ── 1. Loss Curve ──────────────────────────────────────────────────────────
def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    title: str = "Training Loss",
    save_path: str | None = None,
) -> None:
    """Plot training (and optional validation) loss over epochs."""
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, color=PALETTE[0], linewidth=2, label="Train Loss")
    if val_losses:
        plt.plot(val_losses, color=PALETTE[1], linewidth=2,
                 linestyle="--", label="Val Loss")
    plt.title(title, fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 2. Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names or range(len(cm)),
        yticklabels=class_names or range(len(cm)),
    )
    plt.title(title, fontsize=13)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 3. Feature Distributions ───────────────────────────────────────────────
def plot_distributions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Grid of histograms for numeric features."""
    cols = columns or list(df.select_dtypes(include=[np.number]).columns)
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=30, color=PALETTE[i % len(PALETTE)],
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11)
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── 4. Correlation Heatmap ─────────────────────────────────────────────────
def plot_correlation(
    df: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    save_path: str | None = None,
) -> None:
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(max(6, len(corr)), max(5, len(corr) - 1)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    plt.title(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 5. Learning Curve (Bias-Variance) ─────────────────────────────────────
def plot_learning_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Learning Curve",
    cv: int = 5,
    save_path: str | None = None,
) -> None:
    """
    Shows how train and validation score change with dataset size.
    Gap between curves reveals bias/variance trade-off.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color=PALETTE[0], label="Train Score")
    plt.fill_between(train_sizes,
                     train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color=PALETTE[0])
    plt.plot(train_sizes, val_mean, "s--", color=PALETTE[1], label="Val Score")
    plt.fill_between(train_sizes,
                     val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color=PALETTE[1])
    plt.title(title, fontsize=13)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    print("✅ Visualizer Demo\n")

    plot_distributions(df, list(iris.feature_names))
    plot_correlation(df)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plot_confusion_matrix(y_test, y_pred, class_names=list(iris.target_names))
    plot_learning_curve(clf, iris.data, iris.target, title="Random Forest — Learning Curve")
    print("   All plots rendered ✓")
