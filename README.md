# Comparing Machine Learning Methods for Latent Regime Detection in Multivariate Time Series

## Overview

This project studies **latent regime detection** in multivariate time series using a synthetic dataset. We compare geometry-based clustering methods, time-aware state-space models, and elementary machine learning models to understand how different modeling assumptions affect regime recovery.

The main focus is **structural inference and recovery**, not prediction or performance.

---

## 1. Motivation

Many real-world systems generate multivariate time series that exhibit nonstationarity and regime-switching behavior, where the underlying data-generating process changes over time. Examples include physical systems, biological signals, sensor networks, and economic or financial data.

A central challenge is identifying these latent regimes **without strong domain-specific assumptions**. This project investigates how different machine learning models perform at detecting latent regimes in multivariate time series, with an emphasis on **interpretability**, **robustness**, and **methodological comparison** rather than domain-specific forecasting performance.

The primary goal is to understand *which modeling choices matter*, *what kinds of structure each model can recover*, and *how reliably regimes can be identified under noise and correlation*.

---

## 2. Research Questions

* Can unsupervised and weakly supervised machine learning models recover latent regimes in multivariate time series?
* How do classical statistical methods compare to modern ML and shallow deep learning approaches?
* Which features and temporal patterns most strongly influence regime identification?

---

## 3. Synthetic Data

To avoid domain-specific assumptions and provide ground truth, we generate synthetic multivariate time series with:

* Piecewise constant regimes
* Regime-dependent mean and covariance
* Correlated noise across features
* Stochastic regime transitions

We generate a synthetic multivariate time series with:

* **T time steps**
* **d = 4 observed features**
* **3 latent regimes**

Each regime is characterized by a distinct mean and covariance structure. Regimes persist for extended periods before switching, mimicking macroeconomic or financial regimes.

Ground-truth regime labels are retained *only* for evaluation.

---

### Principal Component Analysis (PCA)

We apply **Principal Component Analysis (PCA)** to extract low-dimensional latent factors from the observed features.

Interpretation:

* PCA acts as a proxy for latent macro or market factors
* reduces noise and feature redundancy
* preserves temporal ordering.

---

## 4. Models and Methods

### 4.1 Geometry-Based Baselines

#### KMeans

* Hard clustering in PCA space
* No uncertainty estimates
* No temporal structure

Used as a minimal baseline.

#### Gaussian Mixture Model (GMM)

* Soft clustering via overlapping Gaussian components
* Provides per-point cluster membership probabilities
* Still ignores temporal persistence

GMM performs strongly when regime separation is primarily geometric.

---

### 4.2 Time-Aware Model: Hidden Markov Model (HMM)

We implement a **Gaussian HMM**, which models:

* latent regime persistence via a transition matrix
* regime-specific emission distributions

This introduces an explicit temporal structure absent in clustering models.

#### Observed pathology

Without transition regularization, the HMM may collapse into **oscillatory state sequences**, rapidly alternating between similar states to overfit pointwise noise.

We mitigate this using a **Dirichlet prior on the transition matrix**, which penalizes excessive switching and encourages regime persistence.


### 4.3 Supervised ML Models

We implement three simple Supervised Machine Learning models in order to observe their effectiveness in extracting information from a stochastic time-series without overfitting.

#### Random Forest Classifier

A standard Random Forest Classifier from scikit-learn is implemented. This serves as a baseline comparison for Supervised Machine Learning models.

#### XGBoost

XGBClassifier from XGBoost is implemented. Used primarily for comparison with MLP model to observe whether decision trees or neural networks are more effective for processing time-series data.

### 4.4 Shallow Deep Learning

* Multi-layer Perceptron (MLP) implemented in Sci-kit Learn.

The neural model is intentionally kept shallow to focus on representation learning rather than scale.

---

## 5. Evaluation

We evaluate regime recovery using **Adjusted Rand Index (ARI)**, which:

* is invariant to label permutations
* compares inferred regimes to ground truth

Models are fit and evaluated on the same sequence, as the goal is **latent structure recovery**, not out-of-sample prediction.

Additional metrics used:

* Regime recovery accuracy
* Transition matrix estimation error

---

## 6. Interpretability

Interpretability is a core focus of this project. We analyze:

* Feature importance (tree-based models)
* Permutation importance
* Regime-wise feature statistics
* Estimated transition matrices and regime durations

These analyses help connect learned regimes to observable data characteristics.

---

## 7. Results Summary

* Geometry-based methods recover strong regime structure when emissions are well separated
* GMM outperforms KMeans by modeling uncertainty and heteroskedasticity
* HMMs introduce temporal coherence but require careful regularization
* Increased model flexibility can reduce performance due to overfitting (bias–variance tradeoff)

These behaviors mirror known challenges in economic and financial regime detection.
Across experiments, probabilistic models with explicit state structure perform best at recovering true regimes in synthetic data, while tree-based and neural models provide competitive performance with greater flexibility.

### Takeaway

Regime detection is fundamentally a tradeoff between **geometric separation** and **temporal coherence**. Understanding when and why models fail is as important as raw performance.

---

## 8. Limitations

* Synthetic data may not capture all complexities of real-world systems
* Shallow neural architectures limit representational capacity
* No causal interpretation of regimes is attempted

---

## 9. Future Work

* apply models to real macro or financial data
* incorporate regime-dependent dynamics (e.g. AR-HMM)
* Bayesian HMMs with stronger priors
* Extension to higher-dimensional and irregularly sampled data

---

## 10. Reproducibility

All experiments are fully reproducible. Random seeds are fixed, and all data generation and training scripts are provided. Figures can be regenerated by running the notebooks in order.

---

## 11. Repository Structure

```
latent-regime-ml/
├── configs/
│   ├── easy.yaml
│   ├── medium.yaml
│   ├── hard.yaml
├── notebooks/
│   ├── 01_synthetic_generation.ipynb
│   ├── 02_exploration.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_uml_models.ipynb
│   ├── 05_sml_models.ipynb
│   └── 06_interpretability.ipynb
├── src/
│   ├── data_generation.py
│   ├── evaluation.py
│   ├── models.py
│   └── plotting.py
├── figures/
│   └── ...
└── README.md
```

- `run_experiment.py` — lightweight experiment driver
- `configs/` — example experiment configurations (easy/medium/hard)
- `src/` — core code: data generation, models, evaluation, plotting
- `notebooks/` — exploration, figures, and interactive analyses
- `figures/` — generated artifacts

-------------------------------------------------------------------------------------

## Quickstart

1. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the main experiment script with a config (examples in `configs/`):

```bash
python3 run_experiment.py configs/medium.yaml
```

This prints Adjusted Rand Index (ARI) scores for each method.


## Contact

For questions or suggestions, open an issue or contact the repository owner.