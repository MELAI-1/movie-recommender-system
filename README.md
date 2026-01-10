# 🎬 MovieLens 32M Recommender System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-100%25-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/Numba-JIT_Accelerated-00A3E0?style=flat&logo=numba&logoColor=white)](https://numba.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Performance](https://img.shields.io/badge/Performance-High_Computing-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-black.svg)](https://www.python.org/dev/peps/pep-0008/)

An end-to-end Machine Learning system built **from scratch** to recommend movies using the massive **MovieLens 32M** dataset. This project moves beyond standard libraries, implementing **Alternating Least Squares (ALS)** and **Bayesian Personalized Ranking (BPR)** using pure mathematics and high-performance computing techniques (Sparse Matrices + Numba JIT).

---

## 🎯 Overview

This system is designed to handle **32 million ratings** efficiently on standard hardware. It implements a full pipeline from raw data ingestion to a production-ready interactive dashboard.

### ⚡ Key Technical Features
*   **Massive Scale:** Processes $32 \times 10^6$ ratings using `scipy.sparse` CSR/CSC matrices to minimize memory footprint (reduction from ~76GB to ~300MB).
*   **High Performance:** Custom **Numba** kernels allow training 200k users in seconds by parallelizing CPU operations, bypassing Python's Global Interpreter Lock (GIL).
*   **Cold Start Solved:** Implements a **Feature-Augmented ALS** that projects Genre embeddings into the latent space, allowing predictions for movies with zero ratings.
*   **Ranking Optimization:** Includes **BPR-SGD** for implicit feedback optimization (AUC maximization).
*   **Interactive Dashboard:** A full **Streamlit** web application for real-time recommendations, latent space visualization, and A/B testing simulation.

---

## 🗂 Repository Structure

```text
movie_recsys/
├── config.yaml
├── README.md
├── requirements.txt           # Core dependencies (numpy, numba, streamlit...)
├── data/
│   ├── raw/                   # Place ratings.csv and movies.csv here
│   └── processed/             # Optimized pickle files / sparse matrices
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Efficient Data Ingestion & Indexing
│   ├── eda.py                 # Advanced EDA & Topology Analysis
│   ├── models/
│   │   ├── base.py
│   │   ├── bias_als.py        # Practical 2: Baseline Model
│   │   ├── als_full.py        # Practical 3: Full Matrix Factorization (Numba)
│   │   ├── als_genres.py      # Practical 5: Hybrid Content-Collaborative
│   │   └── bpr.py             # Practical 6: Implicit Feedback SGD
│   ├── evaluation.py          # Metrics (RMSE, Precision@K, NDCG)
│   └── visualization.py       # Publication-ready plotting engine
├── app/
│   └── main.py                # Streamlit Frontend
├── notebooks/                 # Prototyping & Experiments
└── reports/
    ├── figures/               # Generated PDFs for the Academic Report
    └── academic_report.pdf    # Final documentation
