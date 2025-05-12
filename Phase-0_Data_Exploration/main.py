"""
Main script for Twitter sentiment analysis.

This script orchestrates the entire data exploration process.
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add the project root to Python's module search path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import get_logger
from src.data.data_loader import TwitterDataLoader
from src.exploration.text_analysis import TextAnalyzer
from src.exploration.time_analysis import TimeAnalyzer
from src.exploration.user_analysis import UserAnalyzer
from src.config.constants import DATA_DIR

logger = get_logger(__name__)

logger.info("Starting Twitter sentiment analysis data exploration")

def main(save_figures: bool = True, save_data: bool = True, keep_stopwords: bool = False) -> None:
    """
    Run the complete data exploration process.
    
    Args:
        save_figures: Whether to save generated figures
        save_data: Whether to save processed data
        keep_stopwords: Whether to keep stop words in text analysis (default: remove stop words)
    """
    try:
        start_time = datetime.now()
        
        # Load data
        loader = TwitterDataLoader()
        loader.download_dataset()
        df = loader.load_data()
        df = loader.preprocess_data()
        
        # Get basic dataset statistics
        stats = loader.get_dataset_stats()
        logger.info(f"Dataset statistics: {stats['total_tweets']} tweets, {stats['unique_users']} users")
        logger.info(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        logger.info(f"Sentiment distribution: {stats['sentiment_percentages']}")
        
        # Run text analysis
        logger.info("Running text analysis")
        text_analyzer = TextAnalyzer(df)
        text_results = text_analyzer.run_all_analyses(
            save_figures=save_figures, 
            keep_stopwords=keep_stopwords
        )
        
        # Run time analysis
        logger.info("Running time analysis")
        time_analyzer = TimeAnalyzer(df)
        time_results = time_analyzer.run_all_analyses(save_figures=save_figures)
        
        # Run user analysis
        logger.info("Running user analysis")
        user_analyzer = UserAnalyzer(df)
        user_results = user_analyzer.run_all_analyses(save_figures=save_figures)
        
        # Save processed data
        if save_data:
            logger.info("Saving processed data")
            processed_df = df.copy()
            processed_df.to_csv(DATA_DIR / "processed_data.csv", index=False)
            
            # Save word frequencies
            text_results['word_frequencies']['all'].to_csv(DATA_DIR / "token_frequencies.csv", index=False)
            
        # Log execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"Data exploration completed in {execution_time}")
    
    except Exception as e:
        logger.error(f"Error during data exploration: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twitter Sentiment Analysis Data Exploration")
    parser.add_argument("--no-figures", action="store_true", help="Don't save figures")
    parser.add_argument("--no-data", action="store_true", help="Don't save processed data")
    parser.add_argument("--keep-stopwords", action="store_true", help="Keep stop words in text analysis")
    
    args = parser.parse_args()
    
    main(
        save_figures=not args.no_figures, 
        save_data=not args.no_data,
        keep_stopwords=args.keep_stopwords
    )