# Quantifying Prediction Uncertainty in Solar Power Output using Bayesian Neural Networks and Gradient Boosted Regressors

## 📌 Project Objective
This repository contains the programmatic pipeline for forecasting photovoltaic (PV) solar power generation while rigorously quantifying predictive uncertainty. 

Standard machine learning models output point predictions, which fail to communicate risk during volatile weather conditions. This project addresses this by implementing two advanced paradigms:
1. **Bayesian Neural Networks (BNN) via Monte Carlo Dropout:** To quantify **epistemic uncertainty** (model doubt).
2. **Gradient Boosted Quantile Regression (GBQR):** To quantify **aleatoric uncertainty** (inherent data noise) and establish 90% prediction intervals.

This project connects to the global **Climate and Environmental Crises**, providing a framework for resilient power grid management.

---

## 📂 Repository Structure
```text
Solar_Uncertainty_Analysis/
├── data/                   # Directory for processed datasets (ignored in git)
├── models/                 # Directory for exported high-resolution visuals
├── notebooks/              # Jupyter/Colab notebooks for team experimentation
├── src/                
│   ├── data_loader.py      # Automated data acquisition and cleaning pipeline
│   └── model_engine.py     # PyTorch and Scikit-Learn training and evaluation
├── .gitignore              # Protects repository from large data files
├── README.md               # Project documentation and reproducibility instructions
└── requirements.txt        # Python dependency list
