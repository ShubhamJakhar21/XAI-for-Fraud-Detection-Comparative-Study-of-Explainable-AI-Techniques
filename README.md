# 🔍 XAI for Fraud Detection — Comparative Study of Explainable AI Techniques

> A Comparative Study of Explainable AI (XAI) Techniques for Interpreting Machine Learning Models  
> **Chandigarh University | B.E. Computer Science Engineering | May 2026**  
> Authors: Nancy (22BDA70077) · Anmol (22BDA70127) · Shubham (22BDA70135)  
> Supervisor: Dr. Vijay Bhardwaj

---

## 📌 Project Overview

This project implements and compares four prominent **Explainable AI (XAI)** techniques applied to a **fraud detection** machine learning pipeline:

| XAI Technique | Type | Scope |
|---|---|---|
| **SHAP** (Shapley Additive Explanations) | Feature Attribution | Local + Global |
| **LIME** (Local Interpretable Model-Agnostic Explanations) | Feature Attribution | Local |
| **PDP** (Partial Dependence Plots) | Visualization | Global |
| **ICE** (Individual Conditional Expectation) | Visualization | Local |
| **Surrogate Model** (Decision Tree) | Rule-Based | Global |

Four ML models are trained and compared:
- Logistic Regression (interpretable baseline)
- Random Forest (ensemble)
- Gradient Boosting / XGBoost (high-performance)
- Neural Network (MLP — deep learning)

---

## 📁 Project Structure

```
fraud_xai_project/
├── README.md
├── requirements.txt
├── data/
│   └── generate_data.py          # Synthetic fraud dataset generator
├── scripts/
│   ├── 01_data_preparation.py    # Data loading, EDA, preprocessing
│   ├── 02_train_models.py        # Train all 4 ML models
│   ├── 03_xai_shap.py            # SHAP explanations (manual implementation)
│   ├── 04_xai_lime.py            # LIME explanations (manual implementation)
│   ├── 05_xai_pdp_ice.py         # PDP and ICE plots
│   ├── 06_xai_surrogate.py       # Surrogate model explanation
│   └── 07_comparative_analysis.py# Full comparison + evaluation framework
├── notebooks/
│   └── XAI_Fraud_Detection.ipynb # Complete Jupyter notebook (all-in-one)
├── models/                        # Saved trained models (auto-generated)
├── plots/                         # All generated plots (auto-generated)
└── outputs/                       # Reports & tables (auto-generated)
```

---

## ⚙️ Setup & Installation

### Requirements
```
python >= 3.8
scikit-learn
numpy
pandas
matplotlib
seaborn
scipy
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A: Run All Scripts Sequentially
```bash
cd fraud_xai_project

python data/generate_data.py          # Step 0: Generate dataset
python scripts/01_data_preparation.py # Step 1: EDA & preprocessing
python scripts/02_train_models.py     # Step 2: Train ML models
python scripts/03_xai_shap.py         # Step 3: SHAP analysis
python scripts/04_xai_lime.py         # Step 4: LIME analysis
python scripts/05_xai_pdp_ice.py      # Step 5: PDP & ICE plots
python scripts/06_xai_surrogate.py    # Step 6: Surrogate model
python scripts/07_comparative_analysis.py # Step 7: Full comparison
```

### Option B: Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/XAI_Fraud_Detection.ipynb
```

---

## 📊 Evaluation Framework

Techniques are evaluated on 5 criteria (from the paper):

| Criterion | Description |
|---|---|
| **Interpretability** | How easily explanations are understood by users |
| **Fidelity** | How accurately explanations represent model behavior |
| **Stability** | Consistency of explanations under small input changes |
| **Transparency** | Clarity of the explanation process |
| **Practical Utility** | Usefulness in real-world fraud detection |

---

## 📈 Key Results

- Neural Network achieved highest accuracy (~95%) but required XAI for interpretability
- **SHAP** scored highest overall: High Interpretability, High Fidelity, High Stability
- **LIME** is fast and intuitive but shows moderate stability
- **PDP/ICE** best for understanding global feature trends
- **Surrogate models** most accessible for non-technical stakeholders
- No single XAI technique is universally optimal — combining methods is recommended

---

## 📚 References

- Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions
- Ribeiro et al. (2016) — LIME: Why Should I Trust You?
- Molnar (2022) — Interpretable Machine Learning
- Arrieta et al. (2020) — Explainable AI: Concepts, Taxonomies, Opportunities

