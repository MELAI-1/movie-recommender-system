import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os
import warnings

# Configuration for academic-style plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

class MovieLensEDA:
    """
    Advanced Exploratory Data Analysis engine for MovieLens Datasets.
    
    Attributes:
        ratings_path (str): Path to ratings.csv
        movies_path (str): Path to movies.csv
        df_ratings (pd.DataFrame): Loaded ratings dataframe
        df_movies (pd.DataFrame): Loaded movies dataframe
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ratings_path = os.path.join(data_dir, 'ratings.csv')
        self.movies_path = os.path.join(data_dir, 'movies.csv')
        self.df_ratings = None
        self.df_movies = None
        self._output_dir = "figures/eda"
        os.makedirs(self._output_dir, exist_ok=True)

    def load_data(self):
        """Loads data efficiently using specified dtypes to save memory."""
        print(f"[-] Loading data from {self.data_dir}...")
        
        # Optimize types for 25M/32M rows
        dtypes = {
            'userId': 'int32',
            'movieId': 'int32',
            'rating': 'float32',
            'timestamp': 'int64'
        }
        
        self.df_ratings = pd.read_csv(self.ratings_path, dtype=dtypes)
        self.df_movies = pd.read_csv(self.movies_path)
        
        # Convert timestamp to datetime objects
        self.df_ratings['date'] = pd.to_datetime(self.df_ratings['timestamp'], unit='s')
        
        print(f"[+] Data Loaded.")
        print(f"    Ratings: {self.df_ratings.shape}")
        print(f"    Users: {self.df_ratings['userId'].nunique()}")
        print(f"    Items: {self.df_ratings['movieId'].nunique()}")
        
        # Sparsity Calculation
        n_users = self.df_ratings['userId'].nunique()
        n_items = self.df_ratings['movieId'].nunique()
        n_ratings = len(self.df_ratings)
        sparsity = 1 - (n_ratings / (n_users * n_items))
        print(f"    Sparsity: {sparsity:.6%}")

    def plot_rating_distribution(self):
        """Plots the global distribution of ratings (1.0 to 5.0)."""
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x='rating', data=self.df_ratings, palette="viridis")
        ax.set_title("Global Rating Distribution")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count (Millions)")
        self._save_plot("rating_distribution.png")

    def plot_long_tail(self):
        """
        Generates the Log-Log plot of item popularity.
        This detects the Power Law distribution (Scale-Free Network).
        """
        # Group by movie, count ratings, sort descending
        item_counts = self.df_ratings.groupby('movieId').size().sort_values(ascending=False).values
        
        plt.figure(figsize=(10, 6))
        plt.plot(item_counts, color='blue', linewidth=2)
        plt.title("Long Tail Distribution (Item Popularity)")
        plt.xlabel("Item Rank (Sorted by Popularity)")
        plt.ylabel("Number of Ratings")
        plt.xscale('log')
        plt.yscale('log')
        
        # Add theoretical power law line for comparison (visual aid)
        # y = k * x^-alpha
        x = np.arange(1, len(item_counts) + 1)
        # Visual fit: simple inverse proportion for reference
        y_ref = item_counts[0] * (x ** -0.8) 
        plt.plot(x, y_ref, 'r--', label='Power Law Reference ($x^{-0.8}$)')
        
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        self._save_plot("long_tail_log_log.png")

    def plot_user_activity(self):
        """Plots User Activity Distribution (Degree Distribution)."""
        user_counts = self.df_ratings.groupby('userId').size().sort_values(ascending=False).values
        
        plt.figure(figsize=(10, 6))
        plt.plot(user_counts, color='green', linewidth=2)
        plt.title("User Activity Distribution")
        plt.xlabel("User Rank")
        plt.ylabel("Number of Ratings Given")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        self._save_plot("user_activity_log_log.png")

    def plot_temporal_evolution(self):
        """Analyzes how the number of ratings changes over the years."""
        # Resample by month
        temp_df = self.df_ratings.set_index('date')
        monthly_counts = temp_df.resample('M').size()
        
        plt.figure(figsize=(12, 6))
        monthly_counts.plot(color='purple', linewidth=1.5)
        plt.title("Evolution of Number of Ratings Over Time")
        plt.xlabel("Year")
        plt.ylabel("Number of Ratings (Monthly)")
        self._save_plot("temporal_evolution.png")

    def plot_heatmap_sparsity(self, n_sample=100):
        """
        Plots a heatmap of interactions for the top N users and top N items.
        We cannot plot the full 160k x 60k matrix, so we sample the dense head.
        """
        top_users = self.df_ratings['userId'].value_counts().nlargest(n_sample).index
        top_items = self.df_ratings['movieId'].value_counts().nlargest(n_sample).index
        
        # Filter dataframe
        sample_df = self.df_ratings[
            (self.df_ratings['userId'].isin(top_users)) & 
            (self.df_ratings['movieId'].isin(top_items))
        ]
        
        pivot = sample_df.pivot(index='userId', columns='movieId', values='rating')
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot, cmap="YlGnBu", cbar_kws={'label': 'Rating'}, xticklabels=False, yticklabels=False)
        plt.title(f"Interaction Heatmap (Top {n_sample} Users x Top {n_sample} Items)")
        plt.xlabel("Movies (Popular)")
        plt.ylabel("Users (Active)")
        self._save_plot("sparsity_heatmap.png")

    def plot_mean_rating_vs_popularity(self):
        """
        Checks correlation: Do popular movies get higher ratings?
        """
        stats = self.df_ratings.groupby('movieId').agg({'rating': ['count', 'mean']})
        stats.columns = ['count', 'mean']
        
        # Filter out items with very few ratings to reduce noise
        stats = stats[stats['count'] > 50]
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='count', y='mean', data=stats, alpha=0.3, edgecolor=None)
        plt.xscale('log')
        plt.title("Average Rating vs. Popularity (Log Scale)")
        plt.xlabel("Number of Ratings (Log)")
        plt.ylabel("Average Rating")
        
        # Fit a trend line
        z = np.polyfit(np.log(stats['count']), stats['mean'], 1)
        p = np.poly1d(z)
        plt.plot(stats['count'], p(np.log(stats['count'])), "r--", linewidth=2, label="Trend")
        plt.legend()
        
        self._save_plot("correlation_pop_vs_rating.png")

    def _save_plot(self, filename):
        path = os.path.join(self._output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        print(f"    [Saved] {path}")
        plt.close()

    def run_all(self):
        """Executes the full EDA pipeline."""
        self.load_data()
        self.plot_rating_distribution()
        self.plot_long_tail()
        self.plot_user_activity()
        self.plot_temporal_evolution()
        self.plot_heatmap_sparsity()
        self.plot_mean_rating_vs_popularity()
        print("[+] EDA Complete.")