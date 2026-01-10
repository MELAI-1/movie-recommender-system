
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
```

---

## 🔧 Installation & Requirements

**Prerequisites:** Python 3.8+ and a machine with at least 16GB RAM (for the 32M dataset).

```bash
# 1. Clone the repo
git clone https://github.com/MELAI-1/movie-recommender-system.git
cd movie-recommender-system

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🏃‍♂️ How to Run

### 1️⃣ Data Ingestion & EDA
Downloads the dataset, maps IDs to contiguous integers, and generates topological plots.
```bash
python src/data_loader.py --download-all
python src/eda.py
```

### 2️⃣ Train Models
Trains the Bias-Only model and the Full ALS model using Numba acceleration. Performs Hyperparameter Tuning (Random Search).
```bash
# Example: Run the full ALS training script
python src/models/als_full.py
```

### 3️⃣ Launch the App
Interact with the recommender system, view the "Cold Start" demo, and explore the latent space.
```bash
streamlit run app/main.py
```

---

## 🔍 Mathematical Models Implemented

### 1. Feature-Augmented ALS (Hybrid)
We solve the **Cold Start** problem by constraining item vectors $\mathbf{v}_i$ to be close to their genre centroids. The modified loss function is:

$$
J = \sum_{(u,i)} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \tau \sum_{i} ||\mathbf{v}_i - \mathbf{F}^T \mathbf{g}_i||^2 + \lambda ||\Theta||^2
$$

*   **Outcome:** Allows predicting scores for movies like *Alien: Romulus* even if they have 0 ratings, based on the user's affinity for *Sci-Fi*.

### 2. Bayesian Personalized Ranking (BPR)
Optimizes for ranking order rather than rating value using pairwise loss:

$$
\max_{\Theta} \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{ui} - \hat{x}_{uj}) - \lambda_{\Theta} ||\Theta||^2
$$

*   **Outcome:** Superior Precision@10 performance compared to standard ALS.

---

## 📈 Key Results

Results obtained on the MovieLens 32M dataset (temporal split).

| Model Variation | Latent Factors ($K$) | Regularization ($\lambda$) | Test RMSE | Precision@10 |
| :--- | :---: | :---: | :---: | :---: |
| **Bias-Only Baseline** | - | 2.0 | 0.856 | - |
| **Full ALS (Optimized)** | 13 | 5.0 | **0.779** | 0.21 |
| **BPR-MF (Ranking)** | 20 | 0.01 | N/A | **0.24** |

### Visualizations Generated
The system automatically generates academic-grade figures in `reports/figures/`:
*   **Genre Topology:** Heatmaps showing conditional probability of genre co-occurrence.
*   **Latent Space Map:** t-SNE projection of movie vectors revealing semantic clusters (e.g., Horror separated from Children's).
*   **Cold Start Breakdown:** Bar charts explaining *why* a specific movie was recommended based on genre affinity.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss major changes.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Author:** Astride Melvin Fokam Ninyim
**Affiliation:** AIMS South Africa (African Institute for Mathematical Sciences)
**Email:** [melvin@aims.ac.za](mailto:melvin@aims.ac.za)
**LinkedIn:** [Astride Melvin Fokam Ninyim](https://www.linkedin.com/in/astridemelvinfokamninyim11/)
**Project:** [GitHub](https://github.com/MELAI-1/movie-recommender-system)

---

## 🙏 Acknowledgments

*   **GroupLens Research** for the MovieLens dataset.
*   **Prof Ulrich Paquet** (DeepMind/AIMS South Africa) for supervision and course structure.
```
