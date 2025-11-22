
# 🎬 MovieLens Recommendation System – Complete Matrix Factorization Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-black.svg)](https://www.python.org/dev/peps/pep-0008/)

A comprehensive implementation of collaborative filtering techniques for movie recommendation using the MovieLens datasets (1M, 10M, 25M, 32M). Features **ALS with bias terms, full matrix factorization, genre integration, BPR, and A/B testing**, plus a **Streamlit app** for interactive recommendations.

---

## 🎯 Overview

This project implements and evaluates multiple recommendation approaches:

* **ALS with Bias Terms** – Matrix factorization with user/item biases
* **ALS with Latent Factors** – Full matrix factorization with user-item interactions
* **Genre-Integrated ALS** – Incorporates movie genre metadata
* **Bayesian Personalized Ranking (BPR)** – Ranking-based approach for implicit feedback
* **A/B Testing Framework** – Simulated experiments to evaluate model changes
* **Streamlit Application** – Interactive recommendation system for end users

---

## 🗂 Repository Structure

```text
movie_recsys/
├── config.yaml
├── README.md
├── requirements.txt           # numpy, pandas, matplotlib, seaborn, streamlit, numba
├── data/
│   ├── raw/                   # Download  ml-32m here
│   └── processed/             # Saved numpy arrays / pickle files
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Practical 1
│   ├── eda.py                 # Practical 0
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── als.py             # Practical 2, 3, 4, 5
│   │   └── bpr.py             # Practical 6
│   ├── evaluation.py          # Metrics (RMSE, Precision, Recall)
│   └── utils.py               # Logging, A/B Testing simulation
├── app/
│   └── main.py                # Streamlit Application
├── notebooks/                 # Jupyter notebooks for prototyping
└── reports/
    └── figures                # all the figures made for the report 
```

---

## 🔧 Requirements

* **Python** 3.8+
* **Libraries**: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, streamlit

`requirements.txt` example:

```
numpy>=1.19
pandas>=1.3
matplotlib>=3.3
seaborn>=0.11
scikit-learn>=0.24
scipy>=1.7
streamlit>=1.30
```

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/MELAI-1/movie-recommender-system.git
cd movie-recommender-system
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download MovieLens datasets

```bash
python src/data_loader.py --download-all
```

### 4️⃣ Run Exploratory Data Analysis

```bash
python notebooks/01_dataset_analysis.ipynb
```

### 5️⃣ Train Models

```bash
python src/als_bias_only.py
python src/als_full.py
python src/als_with_genres.py
python src/bpr_algorithm.py
```

### 6️⃣ Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Key Dataset Info

**MovieLens 32M**:

* ~32 million ratings
* ~280,000 users
* ~87,000 movies
* Ratings: 0.5 – 5.0 stars
* Timestamps included

---

## 🔍 Model Overview

### Alternating Least Squares (ALS)

Objective function:

```
min_{U,V,b} Σ(r_ui - μ - b_u - b_i - u_u^T v_i)^2 + λ(||U||² + ||V||² + ||b||²)
```

Where:

* `r_ui`: rating from user u for item i
* `μ`: global mean
* `b_u`, `b_i`: user/item bias
* `u_u`, `v_i`: latent vectors
* `λ`: regularization

---

### Genre-Integrated ALS

```
r̂_ui = μ + b_u + b_i + u_u^T v_i + τ * u_u^T g_genres(i)
```

* `g_genres(i)`: genre embedding vector for movie i
* `τ`: genre weight hyperparameter

---

### Bayesian Personalized Ranking (BPR)

* Optimizes ranking via **pairwise comparisons**
* SGD-based implementation for implicit feedback
* Evaluated using precision@k, recall@k, and NDCG@k

---

## 📈 Experimental Results

| Model         | Dataset       | Latent Dims | RMSE (Train) | RMSE (Test) |
| ------------- | ------------- | ----------- | ------------ | ----------- |
| ALS Bias Only | MovieLens 1M  | -           | 0.XXX        | 0.XXX       |
| ALS Full      | MovieLens 1M  | 16          | 0.XXX        | 0.XXX       |
| ALS + Genres  | MovieLens 1M  | 16          | 0.XXX        | 0.XXX       |
| ALS Full      | MovieLens 25M | 16          | 0.XXX        | 0.XXX       |

---

### Visualizations

* **2D Movie Embeddings** – Clusters similar genres
* **Genre Embeddings** – Logical positioning of genres
* **Loss Convergence** – Monotonic decrease during training

---

## 🧪 Reproducibility

* **Hyperparameters**: λ, τ, latent dimensions K, iterations
* **Train/Test Split**: Chronological
* **Cross-validation**: Optional for hyperparameter tuning

---

## 📊 A/B Testing

```python
from src.ab_testing import ABTestFramework

framework = ABTestFramework()
results = framework.run_test(
    model_a=als_baseline,
    model_b=als_with_genres,
    test_users=1000,
    metrics=['precision_at_10', 'ndcg']
)
```

* Statistical significance: Welch’s t-test (α=0.05)

---

## 💻 Streamlit Application

Features:

* User selection / login simulation
* Input ratings and preferences
* Top-N recommendations with movie titles, genres, predicted rating
* Interactive filtering (genres, popularity)
* Toggle between model versions (A/B testing)
* Visualize 2D embeddings for movies and genres
* Log feedback and user interactions

```bash
streamlit run app/streamlit_app.py
```

---

## 🏆 Highlights

* Comprehensive evaluation across MovieLens 1M, 10M, 25M, 32M
* Scalable, production-ready implementation
* Academic-quality report and visualizations
* Scenario-based validation and genre-aware recommendations

---

## 🤝 Contributing

Pull requests welcome. Open issues to discuss major changes.

---

## 📧 Contact

**Author**: Astride Melvin Fokam Ninyim
**Email**: [melvin@aims.ac.za](mailto:melvin@aims.ac.za)
**LinkedIn**: [https://www.linkedin.com/in/astridemelvinfokamninyim11/](https://www.linkedin.com/in/astridemelvinfokamninyim11/)
**Project**: [GitHub](https://github.com/MELAI-1/movie-recommender-system)

---

## 🙏 Acknowledgments

* MovieLens dataset – GroupLens Research, University of Minnesota
* Collaborative filtering research inspiration
* Course supervision: Ulrich Paquet, AIMS South Africa

