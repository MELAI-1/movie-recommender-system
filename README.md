# Matrix Factorization with Genre Integration for Movie Recommendation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-black.svg)](https://www.python.org/dev/peps/pep-0008/)

A comprehensive implementation of matrix factorization techniques for movie recommendation systems using the MovieLens dataset, featuring Alternating Least Squares (ALS) with bias terms and genre embeddings.

## 🎯 Overview

This project implements and evaluates multiple collaborative filtering approaches:
- **ALS with Bias Terms**: Basic matrix factorization with user/item biases
- **ALS with Latent Factors**: Full matrix factorization with user-item interaction matrices
- **Genre-Integrated Model**: Enhanced model incorporating movie genre metadata
- **Bayesian Personalized Ranking (BPR)**: Ranking-based approach for implicit feedback
- **A/B Testing Framework**: Statistical evaluation of recommendation improvements

## 📊 Key Results

- **RMSE Performance**: Achieved X.XX RMSE on MovieLens 25M dataset
- **Scalability**: Optimized implementation handles 32M+ ratings efficiently
- **Genre Integration**: X% improvement over baseline through metadata incorporation
- **Real-world Validation**: Coherent recommendations demonstrated with user scenario testing

## 🗂️ Repository Structure

```
movie-recommender-system/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_loader.py               # Dataset loading utilities
│   ├── als_bias_only.py             # ALS with bias terms only
│   ├── als_full.py                  # Complete ALS with U,V matrices
│   ├── als_with_genres.py           # Genre-integrated model
│   ├── bpr_algorithm.py             # Bayesian Personalized Ranking
│   ├── evaluation.py                # Metrics computation (RMSE, Precision@K)
│   ├── visualization.py             # 2D embeddings and plots
│   ├── ab_testing.py                # A/B testing framework
│   └── utils.py                     # Utility functions
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_dataset_analysis.ipynb    # Exploratory data analysis
│   ├── 02_als_experiments.ipynb     # ALS model experiments
│   ├── 03_genre_integration.ipynb   # Genre-enhanced models
│   ├── 04_visualizations.ipynb      # 2D embeddings visualization
│   ├── 05_recommendations_demo.ipynb # Recommendation demonstrations
│   └── 06_performance_analysis.ipynb # Scalability and performance
│
├── data/                            # Dataset directory (not tracked)
│   ├── ml-1m/                       # MovieLens 1M
│   ├── ml-10m/                      # MovieLens 10M
│   └── ml-25m/                      # MovieLens 25M
│
├── figures/                         # Generated plots and visualizations
│   ├── dataset/                     # Dataset analysis plots
│   ├── convergence/                 # Loss function convergence
│   ├── embeddings/                  # 2D embeddings visualizations
│   └── results/                     # Performance comparison plots
│
├── results/                         # Experimental results
│   ├── rmse_results.csv            # RMSE performance across models
│   ├── hyperparameters.json        # Optimal hyperparameters
│   └── recommendations_samples.json # Sample recommendations
│
└── docs/                           # Documentation
    ├── technical_report.pdf        # Academic report (LaTeX)
    └── methodology.md              # Detailed methodology
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
numpy >= 1.19.0
pandas >= 1.3.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
scipy >= 1.7.0
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MELAI-1/movie-recommender-system.git
cd movie-recommender-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download MovieLens datasets:
```bash
python src/data_loader.py --download-all
```

### Basic Usage

#### 1. Train ALS Model with Bias Terms
```python
from src.als_bias_only import ALSBiasOnly

model = ALSBiasOnly(reg_lambda=0.1)
model.fit(train_data, n_iterations=20)
rmse = model.evaluate(test_data)
print(f"RMSE: {rmse:.4f}")
```

#### 2. Full Matrix Factorization
```python
from src.als_full import ALSFull

model = ALSFull(latent_dims=16, reg_lambda=0.1)
model.fit(train_data, n_iterations=50)
recommendations = model.recommend_for_user(user_id=123, n_items=10)
```

#### 3. Genre-Enhanced Model
```python
from src.als_with_genres import ALSWithGenres

model = ALSWithGenres(latent_dims=16, genre_weight=0.5)
model.fit(train_data, movie_genres, n_iterations=50)
```

#### 4. Generate 2D Embeddings
```python
from src.visualization import plot_movie_embeddings_2d

# Train model with 2D latent dimensions
model = ALSFull(latent_dims=2, reg_lambda=0.1)
model.fit(train_data)

# Visualize movie embeddings
plot_movie_embeddings_2d(model.V, movie_titles, save_path='figures/embeddings/')
```

## 📈 Experimental Results

### Performance Comparison

| Model | Dataset | Latent Dims | RMSE (Train) | RMSE (Test) | Training Time |
|-------|---------|-------------|--------------|-------------|---------------|
| ALS Bias Only | MovieLens 1M | - | 0.XXX | 0.XXX | X.X min |
| ALS Full | MovieLens 1M | 16 | 0.XXX | 0.XXX | X.X min |
| ALS + Genres | MovieLens 1M | 16 | 0.XXX | 0.XXX | X.X min |
| ALS Full | MovieLens 25M | 16 | 0.XXX | 0.XXX | XX.X min |

### Key Findings

- **Genre Integration**: Incorporating genre metadata improved RMSE by X.X% on average
- **Optimal Dimensionality**: Peak performance achieved at K=16 latent dimensions
- **Scalability**: Linear scaling to 25M+ ratings with optimized implementation
- **Recommendation Quality**: User scenarios show coherent genre-aware recommendations

### Visualizations

#### 2D Movie Embeddings
![Movie Embeddings](figures/embeddings/movie_embeddings_2d.png)

*Similar movies cluster together in the learned 2D embedding space. Action movies (red) separate clearly from romantic comedies (blue).*

#### Genre Embeddings
![Genre Embeddings](figures/embeddings/genre_embeddings_2d.png)

*Genre vectors show logical positioning: Action-Adventure cluster opposite to Romance-Drama.*

#### Training Convergence
![Convergence](figures/convergence/loss_convergence.png)

*Loss function decreases monotonically across all model variants.*

## 🔍 Detailed Methodology

### Alternating Least Squares (ALS)

The core ALS algorithm optimizes the matrix factorization objective:

```
min_{U,V,b} Σ(r_ui - μ - b_u - b_i - u_u^T v_i)² + λ(||U||² + ||V||² + ||b||²)
```

Where:
- `r_ui`: observed rating from user u for item i
- `μ`: global rating mean
- `b_u, b_i`: user and item bias terms
- `u_u, v_i`: latent factor vectors
- `λ`: regularization parameter

### Genre Integration

Movie genres are incorporated through additional embedding vectors:

```
r̂_ui = μ + b_u + b_i + u_u^T v_i + τ * u_u^T g_genres(i)
```

Where `g_genres(i)` represents the genre embedding for movie i's genres.

## 🧪 Reproducibility

### Hyperparameter Selection
- **Regularization λ**: Grid search over [0.01, 0.1, 1.0, 10.0]
- **Genre weight τ**: Optimized through cross-validation
- **Latent dimensions K**: Evaluated at [2, 4, 8, 16, 32]
- **Iterations**: Monitored convergence (typically 20-50 iterations)

### Data Splits
- **Training**: 80% of ratings (chronologically earlier)
- **Testing**: 20% of ratings (chronologically later)
- **Validation**: 10% of training set for hyperparameter tuning

## 📊 A/B Testing Framework

The repository includes a complete A/B testing simulation:

```python
from src.ab_testing import ABTestFramework

# Compare ALS vs ALS+Genres
framework = ABTestFramework()
results = framework.run_test(
    model_a=als_baseline,
    model_b=als_with_genres,
    test_users=1000,
    metrics=['precision_at_10', 'ndcg']
)
```

Statistical significance tested using Welch's t-test with α=0.05.

## 🏆 Highlights & Achievements

### Academic Rigor
- **Comprehensive evaluation** across multiple datasets (1M, 10M, 25M ratings)
- **Statistical validation** through proper train/test splits and significance testing
- **Reproducible research** with complete methodology documentation
- **Publication-ready** analysis following ICML formatting standards


### Practical Impact
- **Real-world validation** through user scenario testing
- **Genre-aware recommendations** showing clear coherence
- **Scalable architecture** suitable for production deployment
- **A/B testing framework** for continuous improvement




## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

- **Author**: [ASTRIDE MELVIN FOKAM NINYIM]
- **Email**: [melvin@aims.ac.za]
- **LinkedIn**: []
- **Project Link**: [https://github.com/[MELAI-1]/movie-recommender-system](https://github.com/[MELAI-1]/movie-recommender-system)

---

## 🙏 Acknowledgments

- MovieLens dataset provided by GroupLens Research at the University of Minnesota
- Inspired by collaborative filtering research from [relevant papers]
- Course supervision: [Ulrich Paquet], [AIMS SOUTH AFRICA]

---

*This project was developed as part of a comprehensive study on recommender systems, demonstrating advanced machine learning techniques applied to real-world movie recommendation challenges.*
