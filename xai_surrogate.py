"""
06_xai_surrogate.py
--------------------
Surrogate Model Explanation using a Decision Tree to approximate
the behaviour of complex black-box ML models.

Theory (paper Section 2.3.7):
  A surrogate is a simpler, interpretable model trained to mimic
  the predictions of the black-box model — not the true labels.
  Fidelity measures how well the surrogate matches the black-box.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, os, sys

from sklearn.tree            import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics         import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.makedirs("plots",   exist_ok=True)
os.makedirs("outputs", exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate training
# ─────────────────────────────────────────────────────────────────────────────

def train_surrogate(black_box, X_train, X_test, y_test, feature_names,
                    model_name, max_depth=5):
    """
    Train a Decision Tree surrogate to mimic black_box on X_train.

    Returns
    -------
    surrogate : fitted DecisionTreeClassifier
    metrics   : dict with fidelity, accuracy, f1
    """
    # Pseudo-labels from black-box
    pseudo_labels = black_box.predict(X_train)

    surrogate = DecisionTreeClassifier(
        max_depth=max_depth, random_state=42,
        class_weight="balanced")
    surrogate.fit(X_train, pseudo_labels)

    # Fidelity: agreement between surrogate and black-box on test set
    bb_preds  = black_box.predict(X_test)
    sur_preds = surrogate.predict(X_test)
    fidelity  = accuracy_score(bb_preds, sur_preds)

    # Real accuracy on ground truth
    real_acc  = accuracy_score(y_test, sur_preds)
    real_f1   = f1_score(y_test, sur_preds, zero_division=0)

    metrics = {
        "Model":    model_name,
        "Fidelity": round(fidelity * 100, 2),
        "Accuracy": round(real_acc  * 100, 2),
        "F1-Score": round(real_f1   * 100, 2),
        "Max Depth": max_depth,
    }
    return surrogate, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_surrogate_tree(surrogate, feature_names, model_name):
    """Visualise the surrogate decision tree."""
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(surrogate, feature_names=feature_names,
              class_names=["Legitimate","Fraud"],
              filled=True, rounded=True, fontsize=7,
              ax=ax, impurity=False, proportion=True)
    ax.set_title(f"Surrogate Decision Tree — {model_name}", fontsize=13)
    plt.tight_layout()
    fname = f"plots/06_surrogate_tree_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved {fname}")


def plot_surrogate_feature_importance(surrogate, feature_names, model_name):
    importances = surrogate.feature_importances_
    idx         = np.argsort(importances)
    labels      = [feature_names[i].replace("_"," ").title() for i in idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, importances[idx], color="#ED7D31", edgecolor="white")
    ax.set_xlabel("Gini Importance (from Surrogate Tree)")
    ax.set_title(f"Surrogate Feature Importance — {model_name}")
    plt.tight_layout()
    fname = f"plots/06_surrogate_importance_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(fname)
    plt.close()
    print(f"  [✓] Saved {fname}")


def plot_fidelity_vs_depth(black_box, X_train, X_test, feature_names, model_name):
    """
    Show the fidelity–interpretability trade-off across tree depths.
    (paper Section 4.6: interpretability ↑ fidelity ↓ trade-off)
    """
    depths    = list(range(1, 12))
    fidelities = []
    for d in depths:
        pseudo = black_box.predict(X_train)
        sur    = DecisionTreeClassifier(max_depth=d, random_state=42)
        sur.fit(X_train, pseudo)
        fidelities.append(accuracy_score(black_box.predict(X_test), sur.predict(X_test)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depths, [f*100 for f in fidelities], "o-", color="#4472C4", lw=2)
    ax.axhline(100, color="gray", ls="--", lw=1, label="Perfect fidelity")
    ax.fill_between(depths, [f*100 for f in fidelities], alpha=0.15, color="#4472C4")
    ax.set_xlabel("Tree Depth (Interpretability decreases →)")
    ax.set_ylabel("Fidelity (%)")
    ax.set_title(f"Interpretability vs Fidelity Trade-off — {model_name}")
    ax.legend()
    plt.tight_layout()
    fname = f"plots/06_surrogate_fidelity_depth_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(fname)
    plt.close()
    print(f"  [✓] Saved {fname}")


def print_decision_rules(surrogate, feature_names, model_name, max_depth=3):
    """Print human-readable IF-THEN rules from the surrogate."""
    rules = export_text(surrogate, feature_names=feature_names,
                        max_depth=max_depth, decimals=3)
    print(f"\n  Decision Rules — {model_name} (top {max_depth} levels):\n")
    print(rules[:2000])   # cap output length
    fname = f"outputs/surrogate_rules_{model_name.lower().replace(' ','_')}.txt"
    with open(fname, "w") as f:
        f.write(f"SURROGATE DECISION RULES — {model_name}\n{'='*60}\n\n")
        f.write(rules)
    print(f"  [✓] Full rules saved to {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_surrogate_analysis():
    data          = np.load("data/data_splits.npz")
    feature_names = list(np.load("data/feature_names.npy", allow_pickle=True))
    X_train, X_test = data["X_train"], data["X_test"]
    y_test           = data["y_test"]

    model_files = {
        "Random Forest":    "models/random_forest.pkl",
        "Gradient Boosting":"models/gradient_boosting.pkl",
        "Neural Network":   "models/neural_network.pkl",
    }

    all_metrics = []

    for model_name, fpath in model_files.items():
        if not os.path.exists(fpath):
            print(f"  [!] {fpath} not found.")
            continue

        with open(fpath, "rb") as f:
            black_box = pickle.load(f)

        print(f"\n[Surrogate] Training for {model_name} ...")

        surrogate, metrics = train_surrogate(
            black_box, X_train, X_test, y_test, feature_names, model_name,
            max_depth=5)

        print(f"  Fidelity : {metrics['Fidelity']}%")
        print(f"  Accuracy : {metrics['Accuracy']}%")
        print(f"  F1-Score : {metrics['F1-Score']}%")

        all_metrics.append(metrics)

        plot_surrogate_tree(surrogate, feature_names, model_name)
        plot_surrogate_feature_importance(surrogate, feature_names, model_name)
        plot_fidelity_vs_depth(black_box, X_train, X_test, feature_names, model_name)
        print_decision_rules(surrogate, feature_names, model_name)

    # Summary table
    df = pd.DataFrame(all_metrics)
    print(f"\n{'='*55}")
    print("  SURROGATE MODEL SUMMARY")
    print(f"{'='*55}")
    print(df.to_string(index=False))
    df.to_csv("outputs/surrogate_metrics.csv", index=False)
    print(f"\n[✓] Surrogate summary saved to outputs/surrogate_metrics.csv")
    print("[✓] Surrogate analysis complete.")


if __name__ == "__main__":
    run_surrogate_analysis()
    print("[✓] Step 6 complete — Surrogate done.\n")
