"""
Data Preprocessor — Reusable ML Utilities
==========================================
A clean toolkit for the most common data preparation tasks:
  - Missing value handling
  - Feature normalization / standardization
  - One-hot encoding
  - Train / validation / test split with stratification
  - Outlier detection (IQR method)

Author: Your Name
"""

import numpy as np
import pandas as pd
from typing import Literal


# ── Missing Values ─────────────────────────────────────────────────────────
def fill_missing(
    df: pd.DataFrame,
    strategy: Literal["mean", "median", "mode", "drop"] = "mean",
) -> pd.DataFrame:
    """
    Fill or drop missing values in a DataFrame.

    Parameters
    ----------
    df       : input DataFrame
    strategy : 'mean', 'median', 'mode' fill numeric columns;
               'drop' removes all rows with any NaN.
    """
    df = df.copy()
    if strategy == "drop":
        return df.dropna()

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            if strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)

    # Fill categorical columns with mode
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# ── Normalisation ──────────────────────────────────────────────────────────
def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-Max normalization → scales all features to [0, 1].

    Returns
    -------
    X_norm, min_vals, max_vals  (save min/max to inverse-transform later)
    """
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # avoid division by zero
    X_norm = (X - min_vals) / range_vals
    return X_norm, min_vals, max_vals


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score standardization → mean=0, std=1 per feature.

    Returns
    -------
    X_std, means, stds
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1  # avoid division by zero
    return (X - means) / stds, means, stds


# ── One-Hot Encoding ───────────────────────────────────────────────────────
def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    One-hot encode specified categorical columns.
    Drops the first dummy to avoid multicollinearity.
    """
    return pd.get_dummies(df, columns=columns, drop_first=True)


# ── Outlier Detection ──────────────────────────────────────────────────────
def remove_outliers_iqr(df: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Remove rows where any specified column value is outside
    [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    df = df.copy()
    mask = pd.Series([True] * len(df), index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        mask &= df[col].between(lower, upper)
    removed = (~mask).sum()
    print(f"   Outlier removal: {removed} rows removed ({removed/len(df):.1%})")
    return df[mask]


# ── Train / Val / Test Split ───────────────────────────────────────────────
def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Split data into train / validation / test sets.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_test = int(len(X) * test_size)
    n_val = int(len(X) * val_size)

    test_idx = indices[:n_test]
    val_idx = indices[n_test: n_test + n_val]
    train_idx = indices[n_test + n_val:]

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )


# ── Dataset Summary ────────────────────────────────────────────────────────
def summarise(df: pd.DataFrame) -> None:
    """Print a clean overview of any DataFrame."""
    print(f"\n{'='*50}")
    print(f"  Shape      : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Missing    : {df.isnull().sum().sum()} total NaN values")
    print(f"  Dtypes     :\n{df.dtypes.value_counts().to_string()}")
    print(f"\n  Numeric summary:")
    print(df.describe().round(2).to_string())
    print('='*50)


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic messy dataset
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age":    np.random.randint(18, 65, n).astype(float),
        "income": np.random.normal(50000, 15000, n),
        "score":  np.random.uniform(0, 100, n),
        "city":   np.random.choice(["Delhi", "Mumbai", "Bangalore", None], n),
        "label":  np.random.randint(0, 2, n),
    })
    # Inject some missing values
    df.loc[np.random.choice(n, 15, replace=False), "age"] = np.nan
    df.loc[np.random.choice(n, 10, replace=False), "income"] = np.nan
    # Inject outliers
    df.loc[:5, "income"] = 500_000

    print("✅ Data Preprocessor Demo")
    summarise(df)

    df = fill_missing(df, strategy="median")
    df = remove_outliers_iqr(df, ["income"])
    df = one_hot_encode(df, ["city"])

    X = df.drop("label", axis=1).values
    y = df["label"].values
    X_std, means, stds = standardize(X)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_std, y)
    print(f"\n   Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
    print("\n   Preprocessing pipeline complete ✓")
