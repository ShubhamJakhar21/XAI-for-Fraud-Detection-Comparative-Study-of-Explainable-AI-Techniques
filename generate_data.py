"""
generate_data.py
----------------
Generates a synthetic credit card fraud detection dataset
modelled after real-world fraud patterns.

Features mimic: transaction amount, time, merchant category,
cardholder behaviour, location anomalies, etc.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_fraud_dataset(n_samples=10000, fraud_ratio=0.15, save=True):
    """
    Generate a realistic synthetic fraud detection dataset.

    Returns
    -------
    df_raw : pd.DataFrame  — unscaled features + target
    df_scaled : pd.DataFrame — scaled features + target
    feature_names : list[str]
    """
    n_fraud   = int(n_samples * fraud_ratio)
    n_legit   = n_samples - n_fraud

    # ── Legitimate transactions ──────────────────────────────────────────────
    legit = {
        "transaction_amount":    np.random.exponential(scale=80,  size=n_legit).clip(1, 5000),
        "time_of_day":           np.random.normal(loc=13, scale=4, size=n_legit).clip(0, 23),
        "transaction_frequency": np.random.poisson(lam=5,          size=n_legit).clip(1, 50),
        "account_age_days":      np.random.randint(180, 3650,       size=n_legit),
        "distance_from_home_km": np.random.exponential(scale=15,   size=n_legit).clip(0, 200),
        "merchant_risk_score":   np.random.uniform(0, 0.3,          size=n_legit),
        "num_failed_attempts":   np.random.poisson(lam=0.2,         size=n_legit).clip(0, 10),
        "is_international":      np.random.binomial(1, 0.05,        size=n_legit),
        "credit_utilization":    np.random.uniform(0.1, 0.6,        size=n_legit),
        "avg_transaction_amt":   np.random.normal(loc=75, scale=30, size=n_legit).clip(5, 500),
        "days_since_last_txn":   np.random.exponential(scale=3,     size=n_legit).clip(0, 60),
        "label": np.zeros(n_legit, dtype=int),
    }

    # ── Fraudulent transactions ──────────────────────────────────────────────
    fraud = {
        "transaction_amount":    np.random.exponential(scale=400, size=n_fraud).clip(50, 10000),
        "time_of_day":           np.random.choice(
                                    np.concatenate([np.arange(0, 5), np.arange(22, 24)]),
                                    size=n_fraud),
        "transaction_frequency": np.random.poisson(lam=15,          size=n_fraud).clip(1, 100),
        "account_age_days":      np.random.randint(1, 365,           size=n_fraud),
        "distance_from_home_km": np.random.exponential(scale=200,   size=n_fraud).clip(50, 5000),
        "merchant_risk_score":   np.random.uniform(0.5, 1.0,         size=n_fraud),
        "num_failed_attempts":   np.random.poisson(lam=3,            size=n_fraud).clip(0, 20),
        "is_international":      np.random.binomial(1, 0.6,          size=n_fraud),
        "credit_utilization":    np.random.uniform(0.7, 1.0,         size=n_fraud),
        "avg_transaction_amt":   np.random.normal(loc=75, scale=30,  size=n_fraud).clip(5, 500),
        "days_since_last_txn":   np.random.exponential(scale=0.5,    size=n_fraud).clip(0, 10),
        "label": np.ones(n_fraud, dtype=int),
    }

    df = pd.concat([pd.DataFrame(legit), pd.DataFrame(fraud)], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    feature_names = [c for c in df.columns if c != "label"]

    # ── Scale features ───────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_names])
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    df_scaled["label"] = df["label"].values

    if save:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/fraud_raw.csv",    index=False)
        df_scaled.to_csv("data/fraud_scaled.csv", index=False)
        print(f"[✓] Dataset saved  →  data/fraud_raw.csv  |  data/fraud_scaled.csv")
        print(f"    Total samples  : {len(df)}")
        print(f"    Fraud samples  : {n_fraud} ({fraud_ratio*100:.0f}%)")
        print(f"    Legit samples  : {n_legit}")
        print(f"    Features       : {feature_names}")

    return df, df_scaled, feature_names


if __name__ == "__main__":
    generate_fraud_dataset()
