import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
import yaml
import re
import Wordcloud

# --- Project Root ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# --- Load config.yaml ---
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
try:
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    print("Configuration loaded successfully.")
except FileNotFoundError:
    print(f"Error: Configuration file not found at {CONFIG_PATH}")
    sys.exit(1)

FIGURES_DIR = os.path.join(PROJECT_ROOT, config['paths']['figures'])
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Academic Plotting Configuration ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['pdf.fonttype'] = 42  # Embed fonts (Type 42) for LaTeX compatibility
plt.rcParams['ps.fonttype'] = 42

class MovieLensEDA:
    """
    Advanced Exploratory Data Analysis engine for MovieLens Datasets.
    Includes Power Law detection, Temporal Analysis, and Genre Topology.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ratings_path = os.path.join(data_dir, 'ratings.csv')
        self.movies_path = os.path.join(data_dir, 'movies.csv')
        self.df_ratings = None
        self.df_movies = None
        
        # Output directory for Report Figures
        self._output_dir = "figures/eda"
        os.makedirs(self._output_dir, exist_ok=True)

    def load_data(self):
        """Loads data efficiently using specified dtypes to save memory."""
        print(f"[-] Loading data from {self.data_dir}...")
        
        # Optimize types for 32M rows
        dtypes_ratings = {
            'userId': 'int32',
            'movieId': 'int32',
            'rating': 'float32',
            'timestamp': 'int64'
        }
        
        # Load Ratings
        self.df_ratings = pd.read_csv(self.ratings_path, dtype=dtypes_ratings)
        self.df_ratings['date'] = pd.to_datetime(self.df_ratings['timestamp'], unit='s')
        
        # Load Movies
        self.df_movies = pd.read_csv(self.movies_path, dtype={'movieId': 'int32'})
        
        # Extract Year from Title for temporal analysis: "Toy Story (1995)" -> 1995
        self.df_movies['year'] = self.df_movies['title'].str.extract(r'\((\d{4})\)', expand=False)
        self.df_movies['year'] = pd.to_numeric(self.df_movies['year'], errors='coerce')
        
        print(f"[+] Data Loaded.")
        print(f"    Ratings: {self.df_ratings.shape}")
        print(f"    Users: {self.df_ratings['userId'].nunique()}")
        print(f"    Items: {self.df_ratings['movieId'].nunique()}")

    def plot_rating_distribution(self):
        """Plots the global distribution of ratings."""
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x='rating', data=self.df_ratings, palette="viridis", hue='rating', legend=False)
        ax.set_title("Global Rating Distribution")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count (Millions)")
        self._save_plot("rating_distribution.pdf")

    def plot_long_tail(self):
        """Log-Log plot for Power Law detection (Scale-Free Network check)."""
        item_counts = self.df_ratings.groupby('movieId').size().sort_values(ascending=False).values
        
        plt.figure(figsize=(10, 6))
        # Plot data
        plt.plot(item_counts, color='blue', linewidth=2, label="Actual Data")
        
        # Theoretical Power Law Reference: y = k * x^-alpha
        x = np.arange(1, len(item_counts) + 1)
        alpha = 0.8
        y_ref = item_counts[0] * (x ** -alpha) 
        plt.plot(x, y_ref, 'r--', label=f'Power Law Reference ($\\alpha={alpha}$)')
        
        plt.title("Long Tail Distribution (Item Popularity)")
        plt.xlabel("Item Rank (Log)")
        plt.ylabel("Number of Ratings (Log)")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        self._save_plot("long_tail_log_log.pdf")

    def plot_user_activity(self):
        """Plots User Degree Distribution (Activity levels)."""
        user_counts = self.df_ratings.groupby('userId').size().sort_values(ascending=False).values
        
        plt.figure(figsize=(10, 6))
        plt.plot(user_counts, color='green', linewidth=2)
        plt.title("User Activity Distribution")
        plt.xlabel("User Rank (Log)")
        plt.ylabel("Number of Ratings Given (Log)")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        self._save_plot("user_activity_log_log.pdf")

    def plot_temporal_evolution(self):
        """Time series of ratings activity."""
        temp_df = self.df_ratings.set_index('date')
        monthly_counts = temp_df.resample('ME').size()
        
        plt.figure(figsize=(12, 6))
        monthly_counts.plot(color='purple', linewidth=1.5)
        plt.title("Evolution of Rating Activity Over Time")
        plt.xlabel("Year")
        plt.ylabel("Ratings per Month")
        self._save_plot("temporal_evolution.pdf")

    def plot_movies_per_year(self):
        """Analyzes the content release history."""
        counts = self.df_movies['year'].value_counts().sort_index()
        # Filter reasonable range
        counts = counts[(counts.index > 1920) & (counts.index <= 2023)]
        
        plt.figure(figsize=(12, 6))
        plt.plot(counts.index, counts.values, color='teal', linewidth=2)
        plt.fill_between(counts.index, counts.values, color='teal', alpha=0.1)
        plt.title("Number of Movies Released per Year")
        plt.xlabel("Year")
        plt.ylabel("Count")
        self._save_plot("movies_per_year.pdf")

    def plot_genre_cooccurrence(self):
        """
        Creates a heatmap showing how often genres appear together.
        Uses efficient matrix multiplication for 32M scale.
        """
        # 1. Create Dummy Variables for Genres
        # This creates a (N_movies x N_genres) binary matrix
        genre_dummies = self.df_movies['genres'].str.get_dummies(sep='|')
        
        # 2. Compute Co-occurrence (Adjacency) Matrix: A.T * A
        cooccurrence = genre_dummies.T.dot(genre_dummies)
        
        # 3. Normalize (Jaccard-like or Conditional Probability)
        # Here we normalize by the diagonal (frequency of the genre itself) to see conditional probs
        diag = np.diag(cooccurrence)
        normalized_matrix = cooccurrence.div(diag, axis=0) # P(Column | Row)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(normalized_matrix, cmap="YlGnBu", linewidths=.5, vmin=0, vmax=0.6)
        plt.title("Genre Co-Occurrence Probability (Row Normalized)")
        plt.xlabel("Genre B")
        plt.ylabel("Genre A (Given)")
        self._save_plot("genre_cooccurrence.pdf")

    def plot_mean_rating_vs_popularity(self):
        """Correlation between popularity and average rating."""
        stats = self.df_ratings.groupby('movieId').agg({'rating': ['count', 'mean']})
        stats.columns = ['count', 'mean']
        # Filter for noise
        stats = stats[stats['count'] > 50]
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='count', y='mean', data=stats, alpha=0.3, edgecolor=None)
        
        # Fit Trend Line
        z = np.polyfit(np.log(stats['count']), stats['mean'], 1)
        p = np.poly1d(z)
        plt.plot(stats['count'], p(np.log(stats['count'])), "r--", linewidth=2, label="Log-Linear Trend")
        
        plt.xscale('log')
        plt.title("Average Rating vs. Popularity")
        plt.xlabel("Number of Ratings (Log)")
        plt.ylabel("Average Rating")
        plt.legend()
        self._save_plot("correlation_pop_vs_rating.pdf")

    def plot_wordcloud(self):
        """Generates a WordCloud of movie titles."""
        # Join all titles
        text = " ".join(title for title in self.df_movies['title'].dropna())
        
        wordcloud = WordCloud(width=1600, height=800, background_color='white', max_words=200).generate(text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Most Common Words in Movie Titles")
        self._save_plot("title_wordcloud.pdf")

    def _save_plot(self, filename):
        """
        Saves the current figure strictly as a PDF for academic reports.
        """
        # Enforce PDF extension
        if filename.endswith('.png'):
            filename = filename.replace('.png', '.pdf')
        if not filename.endswith('.pdf'):
            filename += '.pdf'
            
        path = os.path.join(self._output_dir, filename)
        
        plt.tight_layout()
        plt.savefig(path, format='pdf', bbox_inches='tight')
        
        print(f"    [Saved PDF] {path}")
        plt.close()

    def run_all(self):
        """Executes the full EDA pipeline."""
        if self.df_ratings is None:
            self.load_data()
            
        print("[-] Generating Plots...")
        self.plot_rating_distribution()
        self.plot_long_tail()
        self.plot_user_activity()
        self.plot_temporal_evolution()
        self.plot_movies_per_year()
        self.plot_mean_rating_vs_popularity()
        self.plot_genre_cooccurrence()
        self.plot_wordcloud()
        
        print("[+] EDA Complete. All figures saved to figures/eda/")