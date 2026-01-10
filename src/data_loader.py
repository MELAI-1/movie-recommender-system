"""
Data Loading and Preprocessing Module
Efficient data ingestion with optimized dtypes and indexing
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import pickle


class DataLoader:
    """Efficient data loader for MovieLens datasets"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ratings_path = os.path.join(data_dir, 'ratings.csv')
        self.movies_path = os.path.join(data_dir, 'movies.csv')
        self.links_path = os.path.join(data_dir, 'links.csv')
        self.tags_path = os.path.join(data_dir, 'tags.csv')
        
        # Cached data
        self.df_ratings = None
        self.df_movies = None
        self.df_links = None
        self.df_tags = None
        
        # Mappings
        self.userid_to_idx = None
        self.movieid_to_idx = None
        self.idx_to_userid = None
        self.idx_to_movieid = None
    
    def load_ratings(self, optimize_memory: bool = True) -> pd.DataFrame:
        """Load ratings with optimized dtypes"""
        print("[-] Loading ratings...")
        
        if optimize_memory:
            dtypes = {
                'userId': 'int32',
                'movieId': 'int32',
                'rating': 'float32',
                'timestamp': 'int64'
            }
        else:
            dtypes = None
        
        self.df_ratings = pd.read_csv(self.ratings_path, dtype=dtypes)
        
        # Convert timestamp to datetime
        self.df_ratings['date'] = pd.to_datetime(
            self.df_ratings['timestamp'], 
            unit='s'
        )
        
        print(f"[+] Loaded {len(self.df_ratings):,} ratings")
        return self.df_ratings
    
    def load_movies(self) -> pd.DataFrame:
        """Load movies dataset"""
        print("[-] Loading movies...")
        self.df_movies = pd.read_csv(
            self.movies_path, 
            dtype={'movieId': 'int32'}
        )
        print(f"[+] Loaded {len(self.df_movies):,} movies")
        return self.df_movies
    
    def load_links(self) -> pd.DataFrame:
        """Load links dataset"""
        print("[-] Loading links...")
        self.df_links = pd.read_csv(self.links_path)
        return self.df_links
    
    def load_tags(self) -> pd.DataFrame:
        """Load tags dataset"""
        print("[-] Loading tags...")
        self.df_tags = pd.read_csv(self.tags_path)
        return self.df_tags
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all datasets"""
        return (
            self.load_ratings(),
            self.load_movies(),
            self.load_links(),
            self.load_tags()
        )
    
    def create_mappings(self) -> Dict:
        """Create user and movie ID mappings"""
        if self.df_ratings is None:
            self.load_ratings()
        
        print("[-] Creating ID mappings...")
        
        # Get unique sorted IDs
        unique_users = np.unique(self.df_ratings['userId'])
        unique_items = np.unique(self.df_ratings['movieId'])
        
        # Create mappings
        self.userid_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.movieid_to_idx = {mid: i for i, mid in enumerate(unique_items)}
        
        self.idx_to_userid = list(unique_users)
        self.idx_to_movieid = list(unique_items)
        
        # Apply to dataframe
        self.df_ratings['u_idx'] = self.df_ratings['userId'].map(self.userid_to_idx)
        self.df_ratings['i_idx'] = self.df_ratings['movieId'].map(self.movieid_to_idx)
        
        print(f"[+] Mapped {len(unique_users):,} users and {len(unique_items):,} items")
        
        return {
            'userid_to_idx': self.userid_to_idx,
            'movieid_to_idx': self.movieid_to_idx,
            'idx_to_userid': self.idx_to_userid,
            'idx_to_movieid': self.idx_to_movieid,
            'n_users': len(unique_users),
            'n_items': len(unique_items)
        }
    
    def build_adjacency_lists(self) -> Tuple[list, list]:
        """Build list-of-lists structure for fast access"""
        if self.userid_to_idx is None:
            self.create_mappings()
        
        print("[-] Building adjacency lists...")
        
        n_users = len(self.idx_to_userid)
        n_items = len(self.idx_to_movieid)
        
        # Initialize
        data_by_user = [[] for _ in range(n_users)]
        data_by_movie = [[] for _ in range(n_items)]
        
        # Fill both structures
        for u, i, r in zip(
            self.df_ratings['u_idx'], 
            self.df_ratings['i_idx'], 
            self.df_ratings['rating']
        ):
            data_by_user[u].append((i, r))
            data_by_movie[i].append((u, r))
        
        print("[+] Adjacency lists built")
        return data_by_user, data_by_movie
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.df_ratings is None:
            self.load_ratings()
        
        stats = {
            'n_ratings': len(self.df_ratings),
            'n_users': self.df_ratings['userId'].nunique(),
            'n_movies': self.df_ratings['movieId'].nunique(),
            'rating_mean': self.df_ratings['rating'].mean(),
            'rating_std': self.df_ratings['rating'].std(),
            'sparsity': 1 - (len(self.df_ratings) / (
                self.df_ratings['userId'].nunique() * 
                self.df_ratings['movieId'].nunique()
            )),
            'date_range': (
                self.df_ratings['date'].min(),
                self.df_ratings['date'].max()
            )
        }
        
        return stats
    
    def save_processed_data(self, output_dir: str):
        """Save processed data and mappings"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mappings
        mappings = self.create_mappings()
        with open(os.path.join(output_dir, 'mappings.pkl'), 'wb') as f:
            pickle.dump(mappings, f)
        
        # Save processed ratings
        self.df_ratings.to_csv(
            os.path.join(output_dir, 'ratings_processed.csv'),
            index=False
        )
        
        print(f"[+] Processed data saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    loader = DataLoader(data_dir="data/raw/ml-32m")
    
    # Load data
    df_ratings, df_movies, df_links, df_tags = loader.load_all()
    
    # Create mappings
    mappings = loader.create_mappings()
    
    # Build adjacency lists
    data_by_user, data_by_movie = loader.build_adjacency_lists()
    
    # Get statistics
    stats = loader.get_statistics()
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
