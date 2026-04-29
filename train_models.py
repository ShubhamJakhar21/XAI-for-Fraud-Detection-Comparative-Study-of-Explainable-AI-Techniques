"""
02_train_models.py
-------------------
Trains four ML models matching the paper's methodology:
  1. Logistic Regression  — interpretable baseline
  2. Random Forest        — ensemble model
  3. Gradient Boosting    — high-performance black-box
  4. Neural Network (MLP) — deep learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, sys

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics       import (accuracy_score, precision_score, recall_score,
                                   f1_score, roc_auc_score, confusion_matrix,
                                   classification_report, roc_curve)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.makedirs("models",  exist_ok=True)
os.makedirs("plots",   exist_ok=True)
os.makedirs("outputs", exist_ok=True)

RANDOM_SEED = 42
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def load_data():
    data = np.load("data/data_splits.npz")
    feature_names = list(np.load("data/feature_names.npy", allow_pickle=True))
    return (data["X_train"], data["X_val"], data["X_test"],
            data["y_train"], data["y_val"], data["y_test"],
            feature_names)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, C=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=RANDOM_SEED),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            max_iter=500, random_state=RANDOM_SEED, early_stopping=True,
            validation_fraction=0.1, learning_rate_init=0.001),
    }


def evaluate(model, X, y):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y, y_pred)  * 100, 2),
        "Precision": round(precision_score(y, y_pred, zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y, y_pred,    zero_division=0) * 100, 2),
        "F1-Score":  round(f1_score(y, y_pred,        zero_division=0) * 100, 2),
        "ROC-AUC":   round(roc_auc_score(y, y_proba)  * 100, 2),
    }


def plot_results(results_df, models_dict, X_test, y_test):
    # 1. Accuracy bar chart (matching paper Figure 4.1)
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#4472C4", "#70AD47", "#ED7D31", "#9E48B5"]
    bars = ax.bar(results_df["Model"], results_df["Accuracy"],
                  color=colours, edgecolor="white", width=0.5)
    for bar, val in zip(bars, results_df["Accuracy"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylim(70, 100)
    ax.set_title("Accuracy Comparison of Different Machine Learning Models", fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Machine Learning Models")
    plt.tight_layout()
    plt.savefig("plots/02_accuracy_comparison.png")
    plt.close()

    # 2. Multi-metric grouped bar
    metrics  = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x        = np.arange(len(metrics))
    width    = 0.18
    fig, ax  = plt.subplots(figsize=(13, 6))
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.bar(x + i*width, [row[m] for m in metrics],
               width, label=row["Model"], color=colours[i], edgecolor="white")
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(metrics)
    ax.set_ylim(60, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Multi-Metric Performance Comparison of ML Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/02_multi_metric_comparison.png")
    plt.close()

    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, model), colour in zip(models_dict.items(), colours):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colour, lw=2)
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/02_roc_curves.png")
    plt.close()

    # 4. Confusion matrices
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, model), colour in zip(axes, models_dict.items(), colours):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("plots/02_confusion_matrices.png", bbox_inches="tight")
    plt.close()

    print("[✓] Model performance plots saved.")


def train_and_save():
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data()
    X_full_train = np.vstack([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])

    models     = get_models()
    results    = []

    print("\n" + "="*60)
    print("  MODEL TRAINING & EVALUATION")
    print("="*60)

    for name, model in models.items():
        print(f"\n[Training] {name} ...")
        model.fit(X_full_train, y_full_train)
        metrics = evaluate(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        print(f"  Accuracy : {metrics['Accuracy']}%")
        print(f"  F1-Score : {metrics['F1-Score']}%")
        print(f"  ROC-AUC  : {metrics['ROC-AUC']}%")
        # Save model
        with open(f"models/{name.lower().replace(' ','_')}.pkl", "wb") as f:
            pickle.dump(model, f)

    results_df = pd.DataFrame(results)[["Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC"]]
    results_df.to_csv("outputs/model_performance.csv", index=False)
    print(f"\n{'='*60}")
    print(results_df.to_string(index=False))
    print(f"{'='*60}")

    plot_results(results_df, models, X_test, y_test)
    print("[✓] All models saved to models/")
    print("[✓] Performance table saved to outputs/model_performance.csv\n")
    return models, feature_names


if __name__ == "__main__":
    train_and_save()
    print("[✓] Step 2 complete — model training done.\n")
