"""
Iris Flower Classifier — Full sklearn Pipeline
===============================================
End-to-end ML pipeline: preprocessing → model selection
→ hyperparameter tuning → evaluation → feature importance.

Shows how to write production-style ML code with sklearn.

Author: Your Name
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore")


# ── Load & Explore ─────────────────────────────────────────────────────────
def load_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    print(f"\n📊 Dataset: Iris")
    print(f"   Samples  : {X.shape[0]}")
    print(f"   Features : {X.shape[1]} → {feature_names}")
    print(f"   Classes  : {list(class_names)}")
    return X, y, feature_names, class_names


# ── Build Pipelines ────────────────────────────────────────────────────────
def build_pipelines() -> dict:
    """Return a dict of named sklearn Pipelines to compare."""
    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
    }


# ── Grid Search ────────────────────────────────────────────────────────────
def tune_random_forest(X_train, y_train):
    """Hyperparameter search for Random Forest."""
    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 3, 5],
        "clf__min_samples_split": [2, 5],
    }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42)),
    ])
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


# ── Visualisation helpers ──────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig("assets/iris_confusion.png", dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names):
    """Bar chart of feature importances from Random Forest."""
    rf = model.named_steps["clf"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(range(len(importances)),
                   importances[indices],
                   color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"])
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices], rotation=20)
    plt.title("Feature Importances — Random Forest", fontsize=13)
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("assets/feature_importance.png", dpi=150)
    plt.show()


def plot_model_comparison(results: dict):
    """Horizontal bar chart comparing model CV accuracies."""
    names = list(results.keys())
    means = [v["mean"] for v in results.values()]
    stds = [v["std"] for v in results.values()]

    plt.figure(figsize=(7, 3))
    bars = plt.barh(names, means, xerr=stds, color=["#2980b9", "#8e44ad"],
                    edgecolor="white", height=0.4, capsize=5)
    plt.xlim(0.8, 1.02)
    plt.xlabel("5-Fold CV Accuracy")
    plt.title("Model Comparison", fontsize=13)
    for bar, mean in zip(bars, means):
        plt.text(mean + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{mean:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("assets/model_comparison.png", dpi=150)
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, feature_names, class_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compare models with cross-validation
    print("\n🔍 Cross-Validation Comparison")
    pipelines = build_pipelines()
    cv_results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"   {name:20s} → {scores.mean():.4f} ± {scores.std():.4f}")

    plot_model_comparison(cv_results)

    # Hyperparameter tuning for best model
    print("\n⚙️  Tuning Random Forest with GridSearchCV...")
    grid_model = tune_random_forest(X_train, y_train)
    print(f"   Best params : {grid_model.best_params_}")
    print(f"   Best CV acc : {grid_model.best_score_:.4f}")

    # Final evaluation on test set
    best_model = grid_model.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = (y_pred == y_test).mean()

    print(f"\n✅ Final Test Accuracy : {test_acc:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=class_names))

    plot_confusion(y_test, y_pred, class_names)
    plot_feature_importance(best_model, feature_names)
