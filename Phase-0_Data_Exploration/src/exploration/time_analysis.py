"""
Time-based analysis module for Twitter sentiment analysis.

This module focuses on analyzing temporal patterns in tweet sentiment,
such as sentiment distribution by hour, day, or month.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import sys

# Add the project root to the path so we can import modules correctly
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import (
    FIGURE_SIZE_MEDIUM
)
from src.config.constants import (DAYS_OF_WEEK, MONTHS, SENTIMENT_MAP)
from src.utils.plotting import save_figure
from src.utils.logging_utils import get_logger

# Setup logging
logger = get_logger(__name__)

class TimeAnalyzer:
    """
    Class for analyzing temporal patterns in Twitter data.
    
    Analyzes sentiment patterns by hour of day, day of week, and month.
    """
    
    def __init__(self, df: pd.DataFrame, date_col: str = 'Date', 
                target_col: str = 'Target', sentiment_col: str = 'Sentiment'):
        """
        Initialize the time analyzer.
        
        Args:
            df: DataFrame containing the tweets
            date_col: Column name containing the tweet date
            target_col: Column name containing the sentiment label
            sentiment_col: Column name containing the sentiment text
        """
        self.df = df
        self.date_col = date_col
        self.target_col = target_col
        self.sentiment_col = sentiment_col
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_dtype(df[date_col]):
            self.df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Ensure time components are available
        if 'Hour' not in df.columns:
            self.df['Hour'] = self.df[date_col].dt.hour
        if 'DayOfWeek' not in df.columns:
            self.df['DayOfWeek'] = self.df[date_col].dt.day_name()
        if 'Month' not in df.columns:
            self.df['Month'] = self.df[date_col].dt.month_name()
        if 'Year' not in df.columns:
            self.df['Year'] = self.df[date_col].dt.year
            
        # Ensure sentiment mapping is applied
        if sentiment_col not in df.columns:
            self.df[sentiment_col] = self.df[target_col].map(SENTIMENT_MAP)
            
        logger.info(f"TimeAnalyzer initialized with {len(df)} tweets")
        
    def analyze_by_hour(self, save: bool = True) -> pd.DataFrame:
        """
        Analyze sentiment distribution by hour of day.
        
        Args:
            save: Whether to save the generated figure
            
        Returns:
            DataFrame with hourly sentiment counts
        """
        # Group by hour and sentiment
        hourly_sentiment = self.df.groupby(['Hour', self.sentiment_col]).size().unstack().fillna(0)
        
        # Plot
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)
        sns.countplot(data=self.df, x='Hour', hue=self.sentiment_col, palette=["blue", "red"])
        plt.title("Sentiment Distribution by Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Tweet Count")
        plt.legend(title="Sentiment")
        
        if save:
            save_figure("sentiment_by_hour")
        else:
            plt.show()
            
        logger.info(f"Analyzed sentiment by hour")
        return hourly_sentiment
    
    def analyze_by_day(self, save: bool = True) -> pd.DataFrame:
        """
        Analyze sentiment distribution by day of week.
        
        Args:
            save: Whether to save the generated figure
            
        Returns:
            DataFrame with daily sentiment counts
        """
        # Group by day and sentiment
        daily_sentiment = self.df.groupby(['DayOfWeek', self.sentiment_col]).size().unstack().fillna(0)
        
        # Reorder days
        if not daily_sentiment.empty:
            daily_sentiment = daily_sentiment.reindex(DAYS_OF_WEEK)
        
        # Plot
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)
        sns.countplot(
            data=self.df, 
            x='DayOfWeek', 
            hue=self.sentiment_col, 
            order=DAYS_OF_WEEK, 
            palette=["blue", "red"]
        )
        plt.title("Sentiment Distribution by Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Tweet Count")
        plt.legend(title="Sentiment")
        
        if save:
            save_figure("sentiment_by_day")
        else:
            plt.show()
            
        logger.info(f"Analyzed sentiment by day of week")
        return daily_sentiment
    
    def analyze_by_month(self, save: bool = True) -> pd.DataFrame:
        """
        Analyze sentiment distribution by month.
        
        Args:
            save: Whether to save the generated figure
            
        Returns:
            DataFrame with monthly sentiment counts
        """
        # Group by month and sentiment
        monthly_sentiment = self.df.groupby(['Month', self.sentiment_col]).size().unstack().fillna(0)
        
        # Get available months in data
        available_months = sorted(self.df['Month'].unique(), 
                                 key=lambda x: MONTHS.index(x) if x in MONTHS else -1)
        
        # Plot
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)
        sns.countplot(
            data=self.df, 
            x='Month', 
            hue=self.sentiment_col, 
            order=available_months, 
            palette=["blue", "red"]
        )
        plt.title("Sentiment Distribution by Month")
        plt.xlabel("Month")
        plt.ylabel("Tweet Count")
        plt.legend(title="Sentiment")
        plt.xticks(rotation=45)
        
        if save:
            save_figure("sentiment_by_month")
        else:
            plt.show()
            
        logger.info(f"Analyzed sentiment by month")
        return monthly_sentiment
    
    def get_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the time range of the dataset.
        
        Returns:
            Tuple of (oldest_date, newest_date)
        """
        oldest_date = self.df[self.date_col].min()
        newest_date = self.df[self.date_col].max()
        
        logger.info(f"Dataset time range: {oldest_date} to {newest_date}")
        return oldest_date, newest_date
    
    def run_all_analyses(self, save_figures: bool = True) -> Dict[str, Any]:
        """
        Run all time-based analyses and generate visualizations.
        
        Args:
            save_figures: Whether to save generated figures
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Get time range
        oldest_date, newest_date = self.get_time_range()
        results['time_range'] = (oldest_date, newest_date)
        
        # Analyze by hour
        hourly_sentiment = self.analyze_by_hour(save=save_figures)
        results['hourly_sentiment'] = hourly_sentiment
        
        # Analyze by day
        daily_sentiment = self.analyze_by_day(save=save_figures)
        results['daily_sentiment'] = daily_sentiment
        
        # Analyze by month
        monthly_sentiment = self.analyze_by_month(save=save_figures)
        results['monthly_sentiment'] = monthly_sentiment
        
        logger.info("Completed all time-based analyses")
        return results