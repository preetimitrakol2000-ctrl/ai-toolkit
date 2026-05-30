"""
Exploratory Data Analysis — Titanic Dataset
=============================================
A complete EDA notebook (runnable as a script) covering:
  1. Data loading & overview
  2. Missing value analysis
  3. Feature distributions
  4. Survival rate analysis
  5. Correlation study
  6. Key insights summary

Dataset: classic Titanic (via seaborn built-in)

Author: Your Name
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


# ── 1. Load ────────────────────────────────────────────────────────────────
def load_titanic() -> pd.DataFrame:
    df = sns.load_dataset("titanic")
    print(f"\n📦 Titanic Dataset Loaded")
    print(f"   Shape   : {df.shape}")
    print(f"   Columns : {list(df.columns)}")
    return df


# ── 2. Missing Value Heatmap ───────────────────────────────────────────────
def plot_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    plt.figure(figsize=(8, 3.5))
    bars = plt.barh(missing.index, missing.values * 100,
                    color=["#e74c3c" if v > 0.3 else "#f39c12" for v in missing.values])
    plt.axvline(30, color="grey", linestyle="--", alpha=0.6, label="30% threshold")
    plt.xlabel("Missing (%)")
    plt.title("Missing Values by Column", fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/titanic_missing.png", dpi=150)
    plt.show()

    print("\n📉 Missing Values:")
    for col, pct in missing.items():
        flag = "  ⚠️  HIGH" if pct > 0.3 else ""
        print(f"   {col:15s}: {pct:.1%}{flag}")


# ── 3. Survival by Category ────────────────────────────────────────────────
def plot_survival_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # By class
    survival_class = df.groupby("class")["survived"].mean()
    axes[0].bar(survival_class.index, survival_class.values,
                color=["#3498db", "#2ecc71", "#e74c3c"], edgecolor="white")
    axes[0].set_title("Survival Rate by Class")
    axes[0].set_ylabel("Survival Rate")
    axes[0].set_ylim(0, 1)

    # By sex
    survival_sex = df.groupby("sex")["survived"].mean()
    axes[1].bar(survival_sex.index, survival_sex.values,
                color=["#9b59b6", "#f39c12"], edgecolor="white")
    axes[1].set_title("Survival Rate by Sex")
    axes[1].set_ylim(0, 1)

    # By embarked
    survival_emb = df.groupby("embark_town")["survived"].mean().dropna()
    axes[2].bar(survival_emb.index, survival_emb.values,
                color=["#1abc9c", "#e67e22", "#2980b9"], edgecolor="white")
    axes[2].set_title("Survival Rate by Embarkment")
    axes[2].set_ylim(0, 1)

    plt.suptitle("Survival Analysis — Titanic", fontsize=14)
    plt.tight_layout()
    plt.savefig("assets/titanic_survival.png", dpi=150)
    plt.show()


# ── 4. Age Distribution ────────────────────────────────────────────────────
def plot_age_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4))
    for survived, label, color in [(0, "Did Not Survive", "#e74c3c"),
                                   (1, "Survived", "#2ecc71")]:
        subset = df[df["survived"] == survived]["age"].dropna()
        plt.hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor="white")
    plt.axvline(df["age"].median(), color="grey", linestyle="--",
                label=f"Median age: {df['age'].median():.1f}")
    plt.title("Age Distribution by Survival", fontsize=13)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/titanic_age.png", dpi=150)
    plt.show()


# ── 5. Correlation Heatmap ─────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame) -> None:
    numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix — Titanic Features", fontsize=13)
    plt.tight_layout()
    plt.savefig("assets/titanic_correlation.png", dpi=150)
    plt.show()


# ── 6. Key Insights ────────────────────────────────────────────────────────
def print_insights(df: pd.DataFrame) -> None:
    overall = df["survived"].mean()
    female_rate = df[df["sex"] == "female"]["survived"].mean()
    male_rate = df[df["sex"] == "male"]["survived"].mean()
    first_class = df[df["class"] == "First"]["survived"].mean()
    third_class = df[df["class"] == "Third"]["survived"].mean()
    child_rate = df[df["age"] < 12]["survived"].mean()

    print("\n" + "="*52)
    print("  📋  KEY INSIGHTS — TITANIC EDA")
    print("="*52)
    print(f"  Overall survival rate       : {overall:.1%}")
    print(f"  Female survival rate        : {female_rate:.1%}")
    print(f"  Male survival rate          : {male_rate:.1%}")
    print(f"  1st class survival rate     : {first_class:.1%}")
    print(f"  3rd class survival rate     : {third_class:.1%}")
    print(f"  Children (< 12) survival   : {child_rate:.1%}")
    print("="*52)
    print("  → 'Women and children first' policy is visible in data")
    print("  → Passenger class strongly predicts survival")
    print("  → Fare correlates with class (not directly with survival)")
    print("="*52)


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_titanic()
    plot_missing(df)
    plot_survival_analysis(df)
    plot_age_distribution(df)
    plot_correlation(df)
    print_insights(df)
    print("\n✅ EDA complete — all charts saved to assets/")
