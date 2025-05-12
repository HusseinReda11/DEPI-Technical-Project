"""
Data loading module for Twitter sentiment analysis.

This module handles all data loading, preprocessing and basic transformations.
"""
import kagglehub
import pandas as pd
from typing import Optional, Tuple, List
from pathlib import Path
import sys

# Add the project root to the path so we can import modules correctly
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logging_utils import get_logger
from src.config.constants import TWEET_COLUMNS, SENTIMENT_MAP

# Setup logging
logger = get_logger(__name__)

class TwitterDataLoader:
    """
    Class for loading and preparing Twitter data for sentiment analysis.
    
    This class handles downloading data from Kaggle, loading it into pandas,
    and performing initial data preprocessing steps.
    """
    
    def __init__(self, dataset_name: str = "raj713335/twittesentimentanalysis"):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: The name of the Kaggle dataset to download.
        """
        self.dataset_name = dataset_name
        self.data_path: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        
    def download_dataset(self) -> str:
        """
        Download the dataset from Kaggle.
        
        Returns:
            Path to the downloaded dataset.
        """
        logger.info(f"Downloading dataset: {self.dataset_name}")
        self.data_path = kagglehub.dataset_download(self.dataset_name)
        logger.info(f"Dataset downloaded to: {self.data_path}")
        return self.data_path
    
    def load_data(self, file_name: str = "tweets.csv", encoding: str = "ISO-8859-1") -> pd.DataFrame:
        """
        Load the dataset into a pandas DataFrame.
        
        Args:
            file_name: Name of the CSV file to load.
            encoding: Character encoding of the file.
            
        Returns:
            Loaded DataFrame.
        """
        if self.data_path is None:
            self.download_dataset()
            
        file_path = f"{self.data_path}/{file_name}"
        logger.info(f"Loading data from: {file_path}")
        
        self.df = pd.read_csv(
            file_path, 
            encoding=encoding, 
            names=TWEET_COLUMNS, 
            header=None
        )
        
        logger.info(f"Data loaded successfully with {len(self.df)} rows")
        return self.df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Perform initial data preprocessing steps.
        
        Returns:
            Preprocessed DataFrame.
        """
        if self.df is None:
            logger.warning("No data loaded. Loading data first.")
            self.load_data()
            
        logger.info("Preprocessing data")
        
        # Convert Target to category
        self.df['Target'] = self.df['Target'].astype('category')
        
        # Convert Date to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        # Add time-related columns
        self.df["Hour"] = self.df["Date"].dt.hour
        self.df["DayOfWeek"] = self.df["Date"].dt.day_name()
        self.df["Month"] = self.df["Date"].dt.month_name()
        self.df["Year"] = self.df["Date"].dt.year
        
        # Add sentiment text column
        self.df["Sentiment"] = self.df["Target"].map(SENTIMENT_MAP)
        
        logger.info("Data preprocessing completed")
        return self.df
    
    def get_sentiment_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into positive and negative sentiments.
        
        Returns:
            Tuple containing (positive_df, negative_df)
        """
        if self.df is None:
            logger.warning("No data loaded. Loading and preprocessing data first.")
            self.preprocess_data()
            
        positive_df = self.df[self.df['Target'] == 4]
        negative_df = self.df[self.df['Target'] == 0]
        
        logger.info(f"Data split into {len(positive_df)} positive and {len(negative_df)} negative tweets")
        return positive_df, negative_df
    
    def get_dataset_stats(self) -> dict:
        """
        Calculate basic statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics.
        """
        if self.df is None:
            logger.warning("No data loaded. Loading and preprocessing data first.")
            self.preprocess_data()
            
        stats = {
            "total_tweets": len(self.df),
            "unique_users": self.df['User'].nunique(),
            "date_range": (self.df['Date'].min(), self.df['Date'].max()),
            "sentiment_counts": self.df['Target'].value_counts().to_dict(),
            "sentiment_percentages": (self.df['Target'].value_counts(normalize=True) * 100).to_dict()
        }
        
        logger.info(f"Dataset statistics calculated: {len(self.df)} tweets, {stats['unique_users']} unique users")
        return stats