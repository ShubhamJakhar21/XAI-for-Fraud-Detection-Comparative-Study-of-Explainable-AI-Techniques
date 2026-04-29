"""
05_xai_pdp_ice.py
------------------
Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE).

Theory (from paper Section 2.3.6):
  PDP — shows the marginal effect of one feature on model predictions,
        averaging over all other features.
  ICE — shows per-instance lines, revealing heterogeneity that PDP masks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.makedirs("plots",   exist_ok=True)
os.makedirs("outputs", exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# ─────────────────────────────────────────────────────────────────────────────
# PDP / ICE computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pdp_ice(model, X, feature_idx, n_grid=50, n_ice=100):
    """
    Compute PDP and ICE for a single feature.

    Returns
    -------
    grid      : np.ndarray (n_grid,)   — feature values used
    pdp_vals  : np.ndarray (n_grid,)   — average prediction per grid point
    ice_vals  : np.ndarray (n_ice, n_grid) — per-instance predictions
    """
    predict = lambda Z: model.predict_proba(Z)[:, 1]

    grid = np.linspace(X[:, feature_idx].min(),
                       X[:, feature_idx].max(), n_grid)

    # ICE: sample n_ice instances
    ice_idxs = np.random.choice(len(X), min(n_ice, len(X)), replace=False)
    X_ice    = X[ice_idxs].copy()

    ice_vals = np.zeros((len(X_ice), n_grid))
    for j, gval in enumerate(grid):
        X_mod = X_ice.copy()
        X_mod[:, feature_idx] = gval
        ice_vals[:, j] = predict(X_mod)

    pdp_vals = ice_vals.mean(axis=0)
    return grid, pdp_vals, ice_vals


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_pdp(grid, pdp_vals, feat_name, model_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, pdp_vals, color="#4472C4", lw=2.5)
    ax.fill_between(grid, pdp_vals, alpha=0.15, color="#4472C4")
    ax.set_xlabel(feat_name.replace("_"," ").title())
    ax.set_ylabel("Average Predicted Fraud Probability")
    ax.set_title(f"PDP — {feat_name.replace('_',' ').title()} | {model_name}")
    plt.tight_layout()
    fname = (f"plots/05_pdp_{model_name.lower().replace(' ','_')}_"
             f"{feat_name}.png")
    plt.savefig(fname)
    plt.close()
    print(f"  [✓] Saved {fname}")


def plot_ice(grid, ice_vals, pdp_vals, feat_name, model_name, n_lines=80):
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = cm.coolwarm(np.linspace(0.1, 0.9, min(n_lines, len(ice_vals))))
    for i, (row, col) in enumerate(zip(ice_vals[:n_lines], colours)):
        ax.plot(grid, row, color=col, alpha=0.25, lw=0.8)
    ax.plot(grid, pdp_vals, color="black", lw=2.5, label="PDP (mean)", zorder=5)
    ax.set_xlabel(feat_name.replace("_"," ").title())
    ax.set_ylabel("Predicted Fraud Probability")
    ax.set_title(f"ICE + PDP — {feat_name.replace('_',' ').title()} | {model_name}")
    ax.legend()
    plt.tight_layout()
    fname = (f"plots/05_ice_{model_name.lower().replace(' ','_')}_"
             f"{feat_name}.png")
    plt.savefig(fname)
    plt.close()
    print(f"  [✓] Saved {fname}")


def plot_pdp_grid(model, X, feature_names, model_name, top_k=6):
    """
    Grid of PDP subplots for top-k features by variance.
    """
    # Pick features by prediction variance
    importances = []
    predict = lambda Z: model.predict_proba(Z)[:, 1]
    for fi in range(len(feature_names)):
        grid, pdp, _ = compute_pdp_ice(model, X, fi, n_grid=30, n_ice=50)
        importances.append(pdp.std())
    top_feats = np.argsort(importances)[::-1][:top_k]

    ncols = 3
    nrows = (top_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
    axes = axes.flatten()

    for ax, fi in zip(axes, top_feats):
        grid, pdp, ice = compute_pdp_ice(model, X, fi, n_grid=40, n_ice=60)
        for row in ice[:50]:
            ax.plot(grid, row, color="steelblue", alpha=0.15, lw=0.7)
        ax.plot(grid, pdp, color="tomato", lw=2.5, label="PDP")
        ax.set_title(feature_names[fi].replace("_"," ").title(), fontsize=9)
        ax.set_xlabel("Feature value", fontsize=8)
        ax.set_ylabel("Fraud prob", fontsize=8)

    for ax in axes[top_k:]:
        ax.set_visible(False)

    plt.suptitle(f"PDP + ICE Grid — {model_name}", fontsize=13, y=1.01)
    plt.tight_layout()
    fname = f"plots/05_pdp_ice_grid_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_pdp_ice():
    data          = np.load("data/data_splits.npz")
    feature_names = list(np.load("data/feature_names.npy", allow_pickle=True))
    X_test        = data["X_test"]

    model_files = {
        "Random Forest":    "models/random_forest.pkl",
        "Gradient Boosting":"models/gradient_boosting.pkl",
    }

    # Key features to highlight individually
    key_features = [
        "transaction_amount", "merchant_risk_score",
        "distance_from_home_km", "num_failed_attempts"
    ]

    for model_name, fpath in model_files.items():
        if not os.path.exists(fpath):
            print(f"  [!] {fpath} not found — run 02_train_models.py first.")
            continue

        with open(fpath, "rb") as f:
            model = pickle.load(f)

        print(f"\n[PDP/ICE] Computing for {model_name} ...")

        # Individual PDP + ICE for key features
        for feat_name in key_features:
            if feat_name not in feature_names:
                continue
            fi = feature_names.index(feat_name)
            grid, pdp, ice = compute_pdp_ice(model, X_test, fi,
                                             n_grid=50, n_ice=150)
            plot_pdp(grid, pdp, feat_name, model_name)
            plot_ice(grid, ice, pdp, feat_name, model_name)

            # Save PDP data
            df_pdp = pd.DataFrame({"grid_value": grid, "pdp": pdp})
            df_pdp.to_csv(
                f"outputs/pdp_{model_name.lower().replace(' ','_')}_{feat_name}.csv",
                index=False)

        # Grid overview
        plot_pdp_grid(model, X_test, feature_names, model_name, top_k=6)

    print("\n[✓] PDP/ICE analysis complete.")


if __name__ == "__main__":
    run_pdp_ice()
    print("[✓] Step 5 complete — PDP/ICE done.\n")
