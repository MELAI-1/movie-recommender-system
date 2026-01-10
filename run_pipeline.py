#!/usr/bin/env python
"""
MovieLens Recommender System - Main Pipeline
=============================================

Complete end-to-end pipeline for training and evaluating recommendation models.

Usage:
    python run_pipeline.py --mode all
    python run_pipeline.py --mode eda
    python run_pipeline.py --mode train --model bias_als
    python run_pipeline.py --mode evaluate --model full_als
    python run_pipeline.py --mode streamlit

Author: Astride Melvin Fokam Ninyim
Date: January 2025
"""

import argparse
import sys
import os
import yaml
import logging
from pathlib import Path
import pickle
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from eda import MovieLensEDA
from models.bias_als import BiasALS, tune_lambda_bias_als
from models.als_full import MatrixFactorizationALS, grid_search_als
from evaluation import ModelEvaluator
from visualization import PublicationPlotter


def setup_logging(level='INFO'):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_eda(config, logger):
    """Run Exploratory Data Analysis"""
    logger.info("="*60)
    logger.info("PHASE 1: EXPLORATORY DATA ANALYSIS")
    logger.info("="*60)
    
    # Load data
    data_dir = config['data']['raw_dir']
    loader = DataLoader(data_dir)
    
    df_ratings = loader.load_ratings()
    df_movies = loader.load_movies()
    
    # Initialize EDA
    output_dir = os.path.join(config['visualization']['output_dir'], 'eda')
    eda = MovieLensEDA(df_ratings, df_movies, output_dir)
    
    # Run complete analysis
    logger.info("Running comprehensive EDA...")
    eda.run_full_eda()
    
    logger.info(f"✓ EDA complete. Plots saved to {output_dir}")
    
    return df_ratings, df_movies


def prepare_data(config, df_ratings, logger):
    """Prepare train/test splits"""
    logger.info("="*60)
    logger.info("PHASE 2: DATA PREPARATION")
    logger.info("="*60)
    
    from scipy.sparse import csr_matrix
    import numpy as np
    
    # Create mappings
    user_ids = df_ratings['userId'].unique()
    movie_ids = df_ratings['movieId'].unique()
    
    user2idx = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    
    df_ratings['user_idx'] = df_ratings['userId'].map(user2idx)
    df_ratings['movie_idx'] = df_ratings['movieId'].map(movie2idx)
    
    # Temporal split
    df_ratings = df_ratings.sort_values('timestamp')
    split_idx = int(len(df_ratings) * config['data']['train_test_split'])
    
    train_data = df_ratings.iloc[:split_idx]
    test_data = df_ratings.iloc[split_idx:]
    
    logger.info(f"Train size: {len(train_data):,} ratings")
    logger.info(f"Test size: {len(test_data):,} ratings")
    
    # Create sparse matrices
    n_users = len(user_ids)
    n_items = len(movie_ids)
    
    R_train = csr_matrix(
        (train_data['rating'].values,
         (train_data['user_idx'].values, train_data['movie_idx'].values)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    
    R_test = csr_matrix(
        (test_data['rating'].values,
         (test_data['user_idx'].values, test_data['movie_idx'].values)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    
    logger.info(f"Matrix shape: {R_train.shape}")
    logger.info(f"Sparsity: {1 - R_train.nnz/(n_users*n_items):.4%}")
    
    # Save processed data
    processed_dir = config['data']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    
    with open(os.path.join(processed_dir, 'matrices.pkl'), 'wb') as f:
        pickle.dump({
            'R_train': R_train,
            'R_test': R_test,
            'user2idx': user2idx,
            'movie2idx': movie2idx
        }, f)
    
    logger.info(f"✓ Data prepared and saved to {processed_dir}")
    
    return R_train, R_test


def train_bias_als(config, R_train, R_test, logger):
    """Train Bias-Only ALS model"""
    logger.info("="*60)
    logger.info("PHASE 3A: TRAINING BIAS-ONLY ALS")
    logger.info("="*60)
    
    model_config = config['models']['bias_als']
    
    # Hyperparameter tuning
    if config['tuning']['grid_search']['enabled']:
        logger.info("Running hyperparameter tuning...")
        candidates = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_lambda, results = tune_lambda_bias_als(
            R_train, R_test, candidates, n_epochs=5
        )
        model_config['lambda'] = best_lambda
    
    # Train final model
    logger.info(f"Training final model (λ={model_config['lambda']})...")
    model = BiasALS(
        lambda_reg=model_config['lambda'],
        n_epochs=model_config['n_epochs']
    )
    model.fit(R_train, R_test)
    
    # Save model
    model_path = os.path.join(config['data']['processed_dir'], 'bias_als_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'mu': model.mu,
            'b_u': model.b_u,
            'b_i': model.b_i,
            'lambda': model.lambda_reg,
            'train_rmse': model.train_rmse_history,
            'test_rmse': model.test_rmse_history
        }, f)
    
    logger.info(f"✓ Model saved to {model_path}")
    
    # Visualize
    plotter = PublicationPlotter(
        os.path.join(config['visualization']['output_dir'], 'bias_als')
    )
    plotter.plot_convergence(
        model.train_rmse_history,
        model.test_rmse_history,
        title="Bias-Only ALS Convergence",
        filename="convergence"
    )
    
    return model


def train_full_als(config, R_train, R_test, logger):
    """Train Full Matrix Factorization ALS"""
    logger.info("="*60)
    logger.info("PHASE 3B: TRAINING FULL ALS")
    logger.info("="*60)
    
    model_config = config['models']['full_als']
    
    # Hyperparameter tuning
    if config['tuning']['grid_search']['enabled']:
        logger.info("Running grid search...")
        tune_config = config['tuning']['grid_search']
        best_params, results = grid_search_als(
            R_train, R_test,
            k_values=tune_config['n_factors'],
            lambda_values=tune_config['lambda'],
            epochs_per_trial=tune_config['n_epochs']
        )
        model_config['n_factors'] = best_params[0]
        model_config['lambda'] = best_params[1]
    
    # Train final model
    logger.info(f"Training final model (K={model_config['n_factors']}, "
                f"λ={model_config['lambda']})...")
    
    model = MatrixFactorizationALS(
        n_factors=model_config['n_factors'],
        lambda_reg=model_config['lambda'],
        n_epochs=model_config['n_epochs']
    )
    model.fit(R_train, R_test)
    
    # Save model
    model_path = os.path.join(config['data']['processed_dir'], 'full_als_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'U': model.U,
            'V': model.V,
            'b_u': model.b_u,
            'b_i': model.b_i,
            'mu': model.mu,
            'K': model.K,
            'lambda': model.lambda_reg,
            'loss_history': model.loss_history,
            'train_rmse': model.train_rmse_history,
            'test_rmse': model.test_rmse_history
        }, f)
    
    logger.info(f"✓ Model saved to {model_path}")
    
    # Visualize
    plotter = PublicationPlotter(
        os.path.join(config['visualization']['output_dir'], 'full_als')
    )
    plotter.plot_convergence(
        model.train_rmse_history,
        model.test_rmse_history,
        title=f"Full ALS Convergence (K={model.K}, λ={model.lambda_reg})",
        filename="convergence"
    )
    
    return model


def evaluate_model(config, model, R_train, R_test, model_name, logger):
    """Evaluate trained model"""
    logger.info("="*60)
    logger.info(f"PHASE 4: EVALUATING {model_name.upper()}")
    logger.info("="*60)
    
    evaluator = ModelEvaluator(model, R_train, R_test)
    results = evaluator.evaluate_all(k_values=config['evaluation']['k_values'])
    
    evaluator.print_summary(results)
    
    # Save results
    results_path = os.path.join(
        config['reporting']['output_dir'],
        f'{model_name}_evaluation.pkl'
    )
    os.makedirs(config['reporting']['output_dir'], exist_ok=True)
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"✓ Evaluation results saved to {results_path}")
    
    return results


def run_streamlit(config, logger):
    """Launch Streamlit application"""
    logger.info("="*60)
    logger.info("LAUNCHING STREAMLIT APPLICATION")
    logger.info("="*60)
    
    import subprocess
    
    app_path = os.path.join('app', 'main.py')
    port = config['streamlit']['port']
    
    logger.info(f"Starting Streamlit on port {port}...")
    logger.info(f"Access at: http://localhost:{port}")
    
    subprocess.run(['streamlit', 'run', app_path, '--server.port', str(port)])


def main():
    """Main pipeline orchestrator"""
    parser = argparse.ArgumentParser(description='MovieLens Recommender System Pipeline')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['all', 'eda', 'train', 'evaluate', 'streamlit'],
                       help='Pipeline mode')
    
    parser.add_argument('--model', type=str, default='full_als',
                       choices=['bias_als', 'full_als', 'all'],
                       help='Model to train/evaluate')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_level)
    config = load_config(args.config)
    
    logger.info("="*60)
    logger.info("MOVIELENS 32M RECOMMENDER SYSTEM PIPELINE")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("="*60)
    
    try:
        # Phase 1: EDA
        if args.mode in ['all', 'eda']:
            df_ratings, df_movies = run_eda(config, logger)
        else:
            # Load cached data
            data_dir = config['data']['raw_dir']
            loader = DataLoader(data_dir)
            df_ratings = loader.load_ratings()
            df_movies = loader.load_movies()
        
        # Phase 2: Data Preparation
        if args.mode in ['all', 'train', 'evaluate']:
            # Check for cached matrices
            matrices_path = os.path.join(config['data']['processed_dir'], 'matrices.pkl')
            
            if os.path.exists(matrices_path):
                logger.info("Loading cached matrices...")
                with open(matrices_path, 'rb') as f:
                    data = pickle.load(f)
                    R_train = data['R_train']
                    R_test = data['R_test']
            else:
                R_train, R_test = prepare_data(config, df_ratings, logger)
        
        # Phase 3: Training
        if args.mode in ['all', 'train']:
            if args.model in ['bias_als', 'all']:
                bias_model = train_bias_als(config, R_train, R_test, logger)
            
            if args.model in ['full_als', 'all']:
                full_model = train_full_als(config, R_train, R_test, logger)
        
        # Phase 4: Evaluation
        if args.mode in ['all', 'evaluate']:
            # Load models if not just trained
            if args.mode != 'all':
                processed_dir = config['data']['processed_dir']
                
                if args.model in ['bias_als', 'all']:
                    with open(os.path.join(processed_dir, 'bias_als_model.pkl'), 'rb') as f:
                        bias_data = pickle.load(f)
                        bias_model = BiasALS(bias_data['lambda'])
                        bias_model.mu = bias_data['mu']
                        bias_model.b_u = bias_data['b_u']
                        bias_model.b_i = bias_data['b_i']
                
                if args.model in ['full_als', 'all']:
                    with open(os.path.join(processed_dir, 'full_als_model.pkl'), 'rb') as f:
                        full_data = pickle.load(f)
                        full_model = MatrixFactorizationALS(full_data['K'], full_data['lambda'])
                        full_model.U = full_data['U']
                        full_model.V = full_data['V']
                        full_model.b_u = full_data['b_u']
                        full_model.b_i = full_data['b_i']
                        full_model.mu = full_data['mu']
            
            # Evaluate
            if args.model in ['bias_als', 'all']:
                evaluate_model(config, bias_model, R_train, R_test, 'bias_als', logger)
            
            if args.model in ['full_als', 'all']:
                evaluate_model(config, full_model, R_train, R_test, 'full_als', logger)
        
        # Phase 5: Streamlit
        if args.mode == 'streamlit':
            run_streamlit(config, logger)
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
