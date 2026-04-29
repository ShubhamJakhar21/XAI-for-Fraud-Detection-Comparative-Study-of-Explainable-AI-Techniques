"""
04_xai_lime.py
---------------
Manual LIME (Local Interpretable Model-Agnostic Explanations) implementation.

Theory (from paper Section 2.3.4):
  LIME approximates the complex black-box model locally around a specific
  prediction by perturbing the input, getting the model's responses,
  and fitting a simple (ridge regression) interpretable model to those
  perturbed samples — weighted by proximity to the original instance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, os, sys

from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.makedirs("plots",   exist_ok=True)
os.makedirs("outputs", exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# ─────────────────────────────────────────────────────────────────────────────
# Core LIME explainer
# ─────────────────────────────────────────────────────────────────────────────

class LimeExplainer:
    """
    Tabular LIME explainer (continuous features).
    """

    def __init__(self, training_data, feature_names, n_samples=500,
                 kernel_width=0.75, random_state=42):
        self.training_data  = training_data
        self.feature_names  = feature_names
        self.n_samples      = n_samples
        self.kernel_width   = kernel_width
        self.rng            = np.random.RandomState(random_state)
        self.mean_          = training_data.mean(axis=0)
        self.std_           = training_data.std(axis=0) + 1e-8

    def _kernel(self, distances):
        """Exponential kernel — closer samples get higher weight."""
        return np.exp(-(distances ** 2) / (2 * self.kernel_width ** 2))

    def explain_instance(self, instance, predict_fn, n_features=None):
        """
        Explain a single instance.

        Returns
        -------
        coefs : np.ndarray — LIME feature importances
        score : float      — local surrogate R² on perturbed samples
        pred  : float      — model prediction for instance
        """
        n_feat = len(instance)
        if n_features is None:
            n_features = n_feat

        # 1. Perturb around the instance using training distribution
        noise   = self.rng.randn(self.n_samples, n_feat)
        samples = instance + noise * self.std_

        # 2. Get model predictions on perturbed samples
        preds = predict_fn(samples)

        # 3. Compute distances and kernel weights
        dists   = np.sqrt(((samples - instance) ** 2).sum(axis=1)) / n_feat
        weights = self._kernel(dists)

        # 4. Fit interpretable (ridge) model
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(noise, preds, sample_weight=weights)

        # Feature importance = absolute coefficients
        coefs = surrogate.coef_
        score = surrogate.score(noise, preds, sample_weight=weights)
        pred  = predict_fn(instance.reshape(1, -1))[0]

        return coefs, score, pred


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_lime_explanation(coefs, feature_names, pred, score, model_name,
                          instance_idx=0, top_k=10):
    idx    = np.argsort(np.abs(coefs))[::-1][:top_k]
    labels = [feature_names[i].replace("_"," ").title() for i in idx]
    vals   = coefs[idx]
    colours = ["tomato" if v > 0 else "steelblue" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels[::-1], vals[::-1], color=colours[::-1], edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LIME Coefficient (local importance)")
    ax.set_title(
        f"LIME Local Explanation — {model_name}\n"
        f"Instance #{instance_idx} | pred={pred:.3f} | local R²={score:.3f}"
    )
    pos_patch = mpatches.Patch(color="tomato",    label="Pushes toward Fraud")
    neg_patch = mpatches.Patch(color="steelblue", label="Pushes toward Legit")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9)
    plt.tight_layout()
    fname = (f"plots/04_lime_explanation_"
             f"{model_name.lower().replace(' ','_')}_inst{instance_idx}.png")
    plt.savefig(fname)
    plt.close()
    print(f"  [✓] Saved {fname}")


def plot_lime_stability(coefs_list, feature_names, model_name, n_runs=5):
    """
    Show how LIME explanations vary across multiple runs for the same instance.
    Demonstrates the stability limitation mentioned in the paper.
    """
    top_feats = list(np.argsort(np.abs(coefs_list[0]))[::-1][:6])
    labels    = [feature_names[i].replace("_"," ").title() for i in top_feats]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top_feats))
    w = 0.8 / n_runs
    colours = plt.cm.tab10(np.linspace(0, 0.8, n_runs))
    for r, (coefs, col) in enumerate(zip(coefs_list, colours)):
        ax.bar(x + r*w, [coefs[fi] for fi in top_feats],
               w, label=f"Run {r+1}", color=col, alpha=0.85, edgecolor="white")
    ax.set_xticks(x + w * (n_runs/2 - 0.5))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("LIME Coefficient")
    ax.set_title(f"LIME Stability Analysis ({n_runs} runs) — {model_name}\n"
                 f"(Variation shows explanation instability — paper Section 4.3.2)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"plots/04_lime_stability_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(fname)
    plt.close()
    print(f"  [✓] Saved {fname}")


def plot_lime_global(all_coefs, feature_names, model_name):
    """Aggregate LIME over many instances to approximate global importance."""
    mean_abs = np.abs(all_coefs).mean(axis=0)
    idx      = np.argsort(mean_abs)
    fig, ax  = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i].replace("_"," ").title() for i in idx],
            mean_abs[idx], color="#70AD47", edgecolor="white")
    ax.set_xlabel("Mean |LIME Coefficient| across instances")
    ax.set_title(f"LIME Aggregated Global Importance — {model_name}")
    plt.tight_layout()
    fname = f"plots/04_lime_global_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(fname)
    plt.close()
    print(f"  [✓] Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_lime_analysis():
    data          = np.load("data/data_splits.npz")
    feature_names = list(np.load("data/feature_names.npy", allow_pickle=True))
    X_train, X_test = data["X_train"], data["X_test"]
    y_test           = data["y_test"]

    model_files = {
        "Random Forest":    "models/random_forest.pkl",
        "Gradient Boosting":"models/gradient_boosting.pkl",
    }

    for model_name, fpath in model_files.items():
        if not os.path.exists(fpath):
            print(f"  [!] {fpath} not found — run 02_train_models.py first.")
            continue

        with open(fpath, "rb") as f:
            model = pickle.load(f)

        predict_fn = lambda X: model.predict_proba(
            X if X.ndim == 2 else X.reshape(1,-1))[:, 1]

        explainer = LimeExplainer(X_train, feature_names, n_samples=500)

        print(f"\n[LIME] Explaining instances for {model_name} ...")

        # ── Explain several fraud instances ──────────────────────────────────
        fraud_idxs = np.where(y_test == 1)[0][:5]
        all_coefs  = []

        for inst_idx in fraud_idxs:
            coefs, score, pred = explainer.explain_instance(
                X_test[inst_idx], predict_fn)
            all_coefs.append(coefs)
            plot_lime_explanation(coefs, feature_names, pred, score,
                                  model_name, instance_idx=inst_idx)

        # ── Stability: run LIME 5× on same instance ───────────────────────
        if len(fraud_idxs) > 0:
            fi       = fraud_idxs[0]
            runs     = []
            for seed in range(5):
                exp = LimeExplainer(X_train, feature_names,
                                    n_samples=500, random_state=seed*7)
                c, _, _ = exp.explain_instance(X_test[fi], predict_fn)
                runs.append(c)
            plot_lime_stability(runs, feature_names, model_name, n_runs=5)

        # ── Global view ────────────────────────────────────────────────────
        # Explain a larger sample for global aggregation
        sample_idxs = np.random.choice(len(X_test), 30, replace=False)
        g_coefs = []
        for si in sample_idxs:
            c, _, _ = explainer.explain_instance(X_test[si], predict_fn)
            g_coefs.append(c)
        plot_lime_global(np.array(g_coefs), feature_names, model_name)

        # Save
        df_out = pd.DataFrame(all_coefs[:len(fraud_idxs)], columns=feature_names)
        df_out.to_csv(
            f"outputs/lime_coefs_{model_name.lower().replace(' ','_')}.csv",
            index=False)

    print("\n[✓] LIME analysis complete.")


if __name__ == "__main__":
    run_lime_analysis()
    print("[✓] Step 4 complete — LIME done.\n")
