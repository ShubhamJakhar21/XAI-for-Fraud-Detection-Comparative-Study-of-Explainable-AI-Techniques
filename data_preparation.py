"""
01_data_preparation.py
-----------------------
Exploratory Data Analysis (EDA) and data preprocessing
for the fraud detection XAI project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.generate_data import generate_fraud_dataset

os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

RANDOM_SEED = 42
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def run_eda(df_raw, feature_names):
    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(df_raw.describe().round(2).to_string())
    print(f"\nClass distribution:\n{df_raw['label'].value_counts()}")
    print(f"Fraud rate: {df_raw['label'].mean()*100:.1f}%")

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    df_raw["label"].value_counts().plot.bar(ax=ax, color=["steelblue","tomato"], edgecolor="white")
    ax.set_xticklabels(["Legitimate", "Fraud"], rotation=0)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/01_class_distribution.png")
    plt.close()

    # 2. Feature distributions by class
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    for i, feat in enumerate(feature_names):
        for label, colour in [(0, "steelblue"), (1, "tomato")]:
            axes[i].hist(
                df_raw.loc[df_raw["label"] == label, feat],
                bins=40, alpha=0.6, color=colour,
                label="Legit" if label == 0 else "Fraud", density=True
            )
        axes[i].set_title(feat.replace("_", " ").title(), fontsize=9)
        axes[i].legend(fontsize=7)
    # hide unused subplot
    for j in range(len(feature_names), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature Distributions: Fraud vs Legitimate", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("plots/01_feature_distributions.png", bbox_inches="tight")
    plt.close()

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df_raw[feature_names + ["label"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/01_correlation_heatmap.png")
    plt.close()

    print("[✓] EDA plots saved to plots/")


def prepare_data(df_scaled, feature_names, test_size=0.2, val_size=0.1):
    X = df_scaled[feature_names].values
    y = df_scaled["label"].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y_temp)

    print(f"\n[✓] Data split complete:")
    print(f"    Train : {X_train.shape[0]} samples  (fraud: {y_train.sum()})")
    print(f"    Val   : {X_val.shape[0]} samples  (fraud: {y_val.sum()})")
    print(f"    Test  : {X_test.shape[0]} samples  (fraud: {y_test.sum()})")

    # Save splits
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    np.savez("data/data_splits.npz", **splits)
    np.save("data/feature_names.npy", feature_names)
    print("[✓] Splits saved to data/data_splits.npz")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df_raw, df_scaled, feature_names = generate_fraud_dataset(save=True)
    run_eda(df_raw, feature_names)
    prepare_data(df_scaled, feature_names)
    print("\n[✓] Step 1 complete — data preparation done.\n")
