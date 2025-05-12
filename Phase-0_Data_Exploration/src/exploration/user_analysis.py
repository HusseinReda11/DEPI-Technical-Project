"""
User-based analysis module for Twitter sentiment analysis.

This module focuses on analyzing user patterns in tweets, including
top users by frequency and user sentiment patterns.
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
from src.config.config import FIGURE_SIZE_MEDIUM
from src.config.constants import (SENTIMENT_MAP)
from src.utils.plotting import save_figure
from src.utils.logging_utils import get_logger

# Setup logging
logger = get_logger(__name__)

class UserAnalyzer:
    """
    Class for analyzing user patterns in Twitter data.
    
    Analyzes top users by tweet frequency and user sentiment patterns.
    """
    
    def __init__(self, df: pd.DataFrame, user_col: str = 'User', 
                target_col: str = 'Target', sentiment_col: str = 'Sentiment'):
        """
        Initialize the user analyzer.
        
        Args:
            df: DataFrame containing the tweets
            user_col: Column name containing the user identifier
            target_col: Column name containing the sentiment label
            sentiment_col: Column name containing the sentiment text
        """
        self.df = df
        self.user_col = user_col
        self.target_col = target_col
        self.sentiment_col = sentiment_col
        
        # Ensure sentiment mapping is applied
        if sentiment_col not in df.columns:
            self.df[sentiment_col] = self.df[target_col].map(SENTIMENT_MAP)
            
        logger.info(f"UserAnalyzer initialized with {len(df)} tweets from {df[user_col].nunique()} users")
        
    def get_unique_users_count(self) -> int:
        """
        Get the count of unique users in the dataset.
        
        Returns:
            Number of unique users
        """
        unique_users = self.df[self.user_col].nunique()
        logger.info(f"Found {unique_users} unique users")
        return unique_users
    
    def get_top_users(self, top_n: int = 10) -> pd.Series:
        """
        Get the top N users by tweet frequency.
        
        Args:
            top_n: Number of top users to return
            
        Returns:
            Series with user frequencies
        """
        top_users = self.df[self.user_col].value_counts().head(top_n)
        logger.info(f"Retrieved top {top_n} users by frequency")
        return top_users
    
    def plot_top_users(self, top_n: int = 10, save: bool = True) -> pd.Series:
        """
        Plot the top N users by tweet frequency.
        
        Args:
            top_n: Number of top users to plot
            save: Whether to save the generated figure
            
        Returns:
            Series with top user frequencies
        """
        top_users = self.get_top_users(top_n)
        
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)
        sns.barplot(x=top_users.values, y=top_users.index)
        plt.title(f"Top {top_n} Users by Tweet Frequency")
        plt.xlabel("Tweet Count")
        plt.ylabel("User")
        
        if save:
            save_figure("top_users")
        else:
            plt.show()
            
        return top_users
    
    def analyze_user_sentiment(self, top_n: int = 10, save: bool = True) -> pd.DataFrame:
        """
        Analyze sentiment patterns for top users.
        
        Args:
            top_n: Number of top users to analyze
            save: Whether to save the generated figure
            
        Returns:
            DataFrame with user sentiment patterns
        """
        # Group by user and sentiment
        user_sentiment = self.df.groupby([self.user_col, self.sentiment_col]).size().unstack().fillna(0)
        
        # Get top users by total tweets
        top_user_sentiment = user_sentiment.sum(axis=1).sort_values(ascending=False).head(top_n)
        top_user_sentiment_df = user_sentiment.loc[top_user_sentiment.index]
        
        # Plot
        top_user_sentiment_df.plot(kind="barh", figsize=FIGURE_SIZE_MEDIUM, colormap="coolwarm")
        plt.title(f"Sentiment Distribution of Top {top_n} Users")
        plt.xlabel("Tweet Count")
        plt.ylabel("User")
        plt.legend(title="Sentiment")
        
        if save:
            save_figure("top_user_sentiment")
        else:
            plt.show()
            
        logger.info(f"Analyzed sentiment patterns for top {top_n} users")
        return top_user_sentiment_df
    
    def get_user_sentiment_ratios(self) -> pd.DataFrame:
        """
        Calculate sentiment ratios for all users.
        
        Returns:
            DataFrame with user sentiment ratios
        """
        # Group by user and sentiment
        user_sentiment = self.df.groupby([self.user_col, self.sentiment_col]).size().unstack().fillna(0)
        
        # Calculate total tweets per user
        user_sentiment['total'] = user_sentiment.sum(axis=1)
        
        # Calculate sentiment ratios
        for col in user_sentiment.columns:
            if col != 'total':
                user_sentiment[f'{col}_ratio'] = user_sentiment[col] / user_sentiment['total']
                
        logger.info(f"Calculated sentiment ratios for {len(user_sentiment)} users")
        return user_sentiment
    
    def find_polarized_users(self, min_tweets: int = 5) -> Tuple[List[str], List[str]]:
        """
        Find users with strong sentiment polarity (mostly positive or negative).
        
        Args:
            min_tweets: Minimum number of tweets required for consideration
            
        Returns:
            Tuple of (mostly_positive_users, mostly_negative_users)
        """
        # Get user sentiment ratios
        user_sentiment = self.get_user_sentiment_ratios()
        
        # Filter users with minimum number of tweets
        active_users = user_sentiment[user_sentiment['total'] >= min_tweets]
        
        # Identify users with strong polarity
        if 'Positive_ratio' in active_users.columns and 'Negative_ratio' in active_users.columns:
            mostly_positive = active_users[active_users['Positive_ratio'] >= 0.8].index.tolist()
            mostly_negative = active_users[active_users['Negative_ratio'] >= 0.8].index.tolist()
        else:
            positive_col = [col for col in active_users.columns if 'Positive' in col and '_ratio' in col][0]
            negative_col = [col for col in active_users.columns if 'Negative' in col and '_ratio' in col][0]
            mostly_positive = active_users[active_users[positive_col] >= 0.8].index.tolist()
            mostly_negative = active_users[active_users[negative_col] >= 0.8].index.tolist()
        
        logger.info(f"Found {len(mostly_positive)} mostly positive users and {len(mostly_negative)} mostly negative users")
        return mostly_positive, mostly_negative
    
    def run_all_analyses(self, save_figures: bool = True) -> Dict[str, Any]:
        """
        Run all user-based analyses and generate visualizations.
        
        Args:
            save_figures: Whether to save generated figures
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Get unique users count
        unique_users = self.get_unique_users_count()
        results['unique_users'] = unique_users
        
        # Get and plot top users
        top_users = self.plot_top_users(save=save_figures)
        results['top_users'] = top_users
        
        # Analyze user sentiment patterns
        top_user_sentiment = self.analyze_user_sentiment(save=save_figures)
        results['top_user_sentiment'] = top_user_sentiment
        
        # Get user sentiment ratios
        user_sentiment_ratios = self.get_user_sentiment_ratios()
        results['user_sentiment_ratios'] = user_sentiment_ratios
        
        # Find polarized users
        mostly_positive, mostly_negative = self.find_polarized_users()
        results['polarized_users'] = {
            'mostly_positive': mostly_positive,
            'mostly_negative': mostly_negative
        }
        
        logger.info("Completed all user-based analyses")
        return results