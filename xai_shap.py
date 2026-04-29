"""
03_xai_shap.py
---------------
Manual SHAP (SHapley Additive exPlanations) implementation.

Uses permutation-based Shapley value estimation — the model-agnostic
approach that works for ANY sklearn model without the shap library.

Theory (from paper Section 2.3.5):
  SHAP assigns each feature an importance value φᵢ for a particular
  prediction. The sum of all φᵢ equals the difference between the
  model prediction and the mean prediction (baseline).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.makedirs("plots",   exist_ok=True)
os.makedirs("outputs", exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# ─────────────────────────────────────────────────────────────────────────────
# Core SHAP estimator (permutation sampling)
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X_background, X_explain, n_perms=64):
    """
    Estimate Shapley values for X_explain using X_background as baseline.

    Parameters
    ----------
    model        : fitted sklearn model with predict_proba
    X_background : background dataset (subset), shape (m, n_features)
    X_explain    : instances to explain,       shape (k, n_features)
    n_perms      : number of random permutations per instance per feature

    Returns
    -------
    shap_vals : np.ndarray shape (k, n_features)
    base_val  : float — mean prediction on background
    """
    predict = lambda X: model.predict_proba(X)[:, 1]
    base_val = predict(X_background).mean()
    n_feat   = X_explain.shape[1]
    shap_vals = np.zeros((len(X_explain), n_feat))

    for i, x in enumerate(X_explain):
        phi = np.zeros(n_feat)
        for _ in range(n_perms):
            perm  = np.random.permutation(n_feat)
            ref   = X_background[np.random.randint(len(X_background))]
            x_with = ref.copy()
            x_out  = ref.copy()
            for j, feat_idx in enumerate(perm):
                x_with[feat_idx] = x[feat_idx]
                pred_with = predict(x_with.reshape(1, -1))[0]
                pred_out  = predict(x_out.reshape(1, -1))[0]
                phi[feat_idx] += pred_with - pred_out
                x_out[feat_idx] = x[feat_idx]
        shap_vals[i] = phi / n_perms

    return shap_vals, base_val


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_importance(shap_vals, feature_names, model_name, save=True):
    """Bar plot of mean |SHAP| per feature (global importance)."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx      = np.argsort(mean_abs)
    colours  = ["#d73027" if v > 0 else "#4575b4"
                for v in mean_abs[idx]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i].replace("_"," ").title() for i in idx],
            mean_abs[idx], color="#4472C4", edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value|  (mean impact on model output)")
    ax.set_title(f"SHAP Global Feature Importance — {model_name}")
    plt.tight_layout()
    if save:
        fname = f"plots/03_shap_global_{model_name.lower().replace(' ','_')}.png"
        plt.savefig(fname)
        plt.close()
        print(f"  [✓] Saved {fname}")
    else:
        plt.show()


def plot_local_waterfall(shap_vals_single, base_val, feature_names,
                         prediction, model_name, instance_idx=0):
    """Waterfall plot for a single prediction."""
    vals = shap_vals_single
    idx  = np.argsort(np.abs(vals))[::-1][:10]   # top-10 features
    labels = [feature_names[i].replace("_"," ").title() for i in idx]
    values = vals[idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["tomato" if v > 0 else "steelblue" for v in values]
    ax.barh(labels[::-1], values[::-1], color=colours[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title(
        f"SHAP Waterfall — {model_name}\n"
        f"Instance #{instance_idx} | base={base_val:.3f} | pred={prediction:.3f}"
    )
    pos_patch = mpatches.Patch(color="tomato",    label="Increases fraud prob")
    neg_patch = mpatches.Patch(color="steelblue", label="Decreases fraud prob")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9)
    plt.tight_layout()
    fname = (f"plots/03_shap_waterfall_"
             f"{model_name.lower().replace(' ','_')}_inst{instance_idx}.png")
    plt.savefig(fname)
    plt.close()
    print(f"  [✓] Saved {fname}")


def plot_beeswarm(shap_vals, X_explain, feature_names, model_name):
    """Dot-based summary plot (SHAP beeswarm analogue)."""
    n_feat = len(feature_names)
    fig, ax = plt.subplots(figsize=(10, 7))
    for fi in range(n_feat):
        y_jitter = fi + np.random.uniform(-0.3, 0.3, len(shap_vals))
        sc = ax.scatter(shap_vals[:, fi], y_jitter,
                        c=X_explain[:, fi], cmap="coolwarm",
                        alpha=0.4, s=8, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Feature value (low → high)")
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([f.replace("_"," ").title() for f in feature_names], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value")
    ax.set_title(f"SHAP Summary (Beeswarm) — {model_name}")
    plt.tight_layout()
    fname = f"plots/03_shap_beeswarm_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(fname)
    plt.close()
    print(f"  [✓] Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis():
    data         = np.load("data/data_splits.npz")
    feature_names = list(np.load("data/feature_names.npy", allow_pickle=True))
    X_train, X_test = data["X_train"], data["X_test"]
    y_test          = data["y_test"]

    # Use a small background for speed
    bg_idx  = np.random.choice(len(X_train), 100, replace=False)
    X_bg    = X_train[bg_idx]
    X_exp   = X_test[:50]                          # explain 50 test instances

    model_files = {
        "Random Forest":    "models/random_forest.pkl",
        "Gradient Boosting":"models/gradient_boosting.pkl",
    }

    all_results = {}

    for model_name, fpath in model_files.items():
        if not os.path.exists(fpath):
            print(f"  [!] {fpath} not found — run 02_train_models.py first.")
            continue

        with open(fpath, "rb") as f:
            model = pickle.load(f)

        print(f"\n[SHAP] Computing values for {model_name} ...")
        shap_vals, base_val = compute_shap_values(model, X_bg, X_exp, n_perms=64)
        all_results[model_name] = {"shap_vals": shap_vals, "base_val": base_val}

        # Global importance
        plot_global_importance(shap_vals, feature_names, model_name)

        # Local waterfall for fraud instance
        fraud_idxs = np.where(y_test[:50] == 1)[0]
        if len(fraud_idxs) > 0:
            fi = fraud_idxs[0]
            pred = model.predict_proba(X_exp[fi].reshape(1,-1))[0, 1]
            plot_local_waterfall(shap_vals[fi], base_val,
                                 feature_names, pred, model_name, fi)

        # Beeswarm
        plot_beeswarm(shap_vals, X_exp, feature_names, model_name)

        # Save values
        df_out = pd.DataFrame(shap_vals, columns=feature_names)
        df_out["base_value"] = base_val
        df_out.to_csv(f"outputs/shap_values_{model_name.lower().replace(' ','_')}.csv",
                      index=False)

    print("\n[✓] SHAP analysis complete.")
    return all_results


if __name__ == "__main__":
    run_shap_analysis()
    print("[✓] Step 3 complete — SHAP done.\n")
