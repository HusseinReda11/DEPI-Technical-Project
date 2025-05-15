"""
Text analysis module for Twitter sentiment analysis.

This module focuses on text content analysis, such as word frequencies,
n-grams, and language feature analysis.
"""
import pandas as pd
import numpy as np
from collections import Counter
from nltk import ngrams
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import nltk
from nltk.corpus import stopwords

# Download nltk resources if not available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Add the project root to the path so we can import modules correctly
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.constants import (
    WORD_PATTERN, HASHTAG_PATTERN, MENTION_PATTERN, 
    URL_PATTERN, PUNCT_PATTERNS
)
from src.utils.plotting import (
    plot_top_words, plot_sentiment_word_comparison, 
    generate_wordcloud, save_figure
)
from src.utils.logging_utils import get_logger

# setup logging
logger = get_logger(__name__)

class TextAnalyzer:
    """
    Class for analyzing text content in tweets.
    
    Handles word frequency analysis, n-grams, hashtags, mentions,
    and other language features.
    """
    
    def __init__(self, df: pd.DataFrame, text_col: str = 'Text', 
                target_col: str = 'Target'):
        """
        Initialize the text analyzer.
        
        Args:
            df: DataFrame containing the tweets
            text_col: Column name containing the tweet text
            target_col: Column name containing the sentiment label
        """
        self.df = df
        self.text_col = text_col
        self.target_col = target_col
        self.positive_df = df[df[target_col] == 4]
        self.negative_df = df[df[target_col] == 0]
        self.stopwords = set(stopwords.words('english'))
        logger.info(f"TextAnalyzer initialized with {len(df)} tweets")
        
    def extract_hashtags(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract hashtags from tweets.
        
        Args:
            df: DataFrame to analyze (uses self.df if None)
            
        Returns:
            DataFrame with hashtag frequencies
        """
        df = df if df is not None else self.df
        
        hashtag_freq = df[self.text_col].str.lower().str.findall(HASHTAG_PATTERN).explode().value_counts()
        
        # Convert to DataFrame
        hashtag_freq_df = hashtag_freq.reset_index()
        hashtag_freq_df.columns = ['word', 'count']
        
        logger.info(f"Extracted {len(hashtag_freq_df)} unique hashtags")
        return hashtag_freq_df

    def extract_mentions(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract mentions (@user) from tweets.
        
        Args:
            df: DataFrame to analyze (uses self.df if None)
            
        Returns:
            DataFrame with mention frequencies
        """
        df = df if df is not None else self.df
        
        mention_freq = df[self.text_col].str.lower().str.findall(MENTION_PATTERN).explode().value_counts()
        
        # Convert to DataFrame
        mention_freq_df = mention_freq.reset_index()
        mention_freq_df.columns = ['word', 'count']
        
        logger.info(f"Extracted {len(mention_freq_df)} unique mentions")
        return mention_freq_df

    def analyze_punctuation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze punctuation usage across sentiments.
        
        Returns:
            Tuple containing (positive_punct_df, negative_punct_df)
        """
        # For positive sentiment
        positive_punct_counts = {}
        positive_tweets = self.positive_df[self.text_col]
        positive_tweet_count = len(positive_tweets)

        for punct_name, pattern in PUNCT_PATTERNS.items():
            positive_punct_counts[punct_name] = positive_tweets.str.count(pattern).sum()

        positive_punct_freq_df = pd.DataFrame(
            list(positive_punct_counts.items()), 
            columns=['punctuation', 'count']
        )
        positive_punct_freq_df['per_tweet'] = positive_punct_freq_df['count'] / positive_tweet_count

        # For negative sentiment
        negative_punct_counts = {}
        negative_tweets = self.negative_df[self.text_col]
        negative_tweet_count = len(negative_tweets)

        for punct_name, pattern in PUNCT_PATTERNS.items():
            negative_punct_counts[punct_name] = negative_tweets.str.count(pattern).sum()

        negative_punct_freq_df = pd.DataFrame(
            list(negative_punct_counts.items()), 
            columns=['punctuation', 'count']
        )
        negative_punct_freq_df['per_tweet'] = negative_punct_freq_df['count'] / negative_tweet_count
        
        logger.info("Analyzed punctuation usage across sentiments")
        return positive_punct_freq_df, negative_punct_freq_df

    def analyze_urls(self) -> Dict[str, int]:
        """
        Analyze URL usage across sentiments.
        
        Returns:
            Dictionary with URL counts for positive and negative tweets
        """
        url_positive_freq = len(self.positive_df[self.text_col].str.lower().str.findall(URL_PATTERN).explode())
        url_negative_freq = len(self.negative_df[self.text_col].str.lower().str.findall(URL_PATTERN).explode())
        
        url_stats = {
            "positive_urls": url_positive_freq,
            "negative_urls": url_negative_freq
        }
        
        logger.info(f"Analyzed URL usage: {url_positive_freq} in positive, {url_negative_freq} in negative")
        return url_stats

    def analyze_capitalization(self) -> Dict[str, float]:
        """
        Analyze capitalization usage across sentiments.
        
        Returns:
            Dictionary with capitalization rates for positive and negative tweets
        """
        caps_positive_freq = len(self.positive_df[self.text_col].str.findall(r'[A-Z]').explode()) / \
                            len(self.positive_df[self.text_col].str.findall(r'[a-zA-Z]').explode())
        
        caps_negative_freq = len(self.negative_df[self.text_col].str.findall(r'[A-Z]').explode()) / \
                            len(self.negative_df[self.text_col].str.findall(r'[a-zA-Z]').explode())
        
        caps_stats = {
            "positive_caps_rate": caps_positive_freq,
            "negative_caps_rate": caps_negative_freq
        }
        
        logger.info(f"Analyzed capitalization usage: {caps_positive_freq:.4f} in positive, {caps_negative_freq:.4f} in negative")
        return caps_stats

    def plot_punctuation_comparison(self, pos_df: pd.DataFrame, neg_df: pd.DataFrame, 
                                save: bool = True) -> None:
        """
        Plot punctuation usage comparison between positive and negative tweets.
        
        Args:
            pos_df: DataFrame with punctuation data for positive tweets
            neg_df: DataFrame with punctuation data for negative tweets
            save: Whether to save the figure
        """
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Set positions for bars
        x = np.arange(len(pos_df))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, pos_df['per_tweet'], width, 
                label='Positive', color='green', alpha=0.6)
        plt.bar(x + width/2, neg_df['per_tweet'], width, 
                label='Negative', color='red', alpha=0.6)
        
        # Customize plot
        plt.xlabel('Punctuation Type')
        plt.ylabel('Average Usage per Tweet')
        plt.title('Punctuation Usage in Positive vs Negative Tweets')
        plt.xticks(x, pos_df['punctuation'], rotation=45)
        plt.legend()
        
        # Add value labels
        for i, v in enumerate(pos_df['per_tweet']):
            plt.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
        for i, v in enumerate(neg_df['per_tweet']):
            plt.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            save_figure("punctuation_comparison")
        else:
            plt.show()

    def extract_word_frequencies(self, df: Optional[pd.DataFrame] = None, 
                               pattern: str = WORD_PATTERN,
                               keep_stopwords: bool = False) -> pd.DataFrame:
        """
        Extract word frequencies from tweet text.
        
        Args:
            df: DataFrame to analyze (uses self.df if None)
            pattern: Regex pattern to extract words
            keep_stopwords: Whether to keep stop words
            
        Returns:
            DataFrame with word frequencies
        """
        df = df if df is not None else self.df
        words = df[self.text_col].str.lower().str.findall(pattern).explode()
        
        # Remove stop words if needed
        if not keep_stopwords:
            words = words[~words.isin(self.stopwords)]
            logger.info("Removed stop words from word frequencies")
        
        word_freq = words.value_counts()
        
        # Convert to DataFrame
        word_freq_df = word_freq.reset_index()
        word_freq_df.columns = ['word', 'count']
        
        logger.info(f"Extracted {len(word_freq_df)} unique words")
        return word_freq_df
    
    def analyze_all_word_frequencies(self, keep_stopwords: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Analyze word frequencies for all tweets and by sentiment.
        
        Args:
            keep_stopwords: Whether to keep stop words
            
        Returns:
            Tuple containing (all_words_df, positive_words_df, negative_words_df)
        """
        # Get word frequencies for all tweets
        all_words_df = self.extract_word_frequencies(
            self.df, keep_stopwords=keep_stopwords
        )
        
        # Get word frequencies for positive tweets
        positive_words_df = self.extract_word_frequencies(
            self.positive_df, keep_stopwords=keep_stopwords
        )
        
        # Get word frequencies for negative tweets
        negative_words_df = self.extract_word_frequencies(
            self.negative_df, keep_stopwords=keep_stopwords
        )
        
        logger.info(f"Analyzed word frequencies: {len(all_words_df)} total, "
                f"{len(positive_words_df)} positive, {len(negative_words_df)} negative")
        
        return all_words_df, positive_words_df, negative_words_df

    def analyze_ngrams(self, n: int = 2, top_k: int = 20, 
                    df: Optional[pd.DataFrame] = None,
                    keep_stopwords: bool = False) -> pd.DataFrame:
        """
        Analyze n-grams in tweet text.
        
        Args:
            n: Size of n-grams to analyze
            top_k: Number of top n-grams to return
            df: DataFrame to analyze (uses self.df if None)
            keep_stopwords: Whether to keep stop words
            
        Returns:
            DataFrame with n-gram frequencies
        """
        df = df if df is not None else self.df
        
        # Preprocess and generate n-grams
        ngram_counts = Counter()
        
        for text in df[self.text_col]:
            # Convert to lowercase and tokenize
            tokens = str(text).lower().split()
            
            # Remove stop words if needed
            if not keep_stopwords:
                tokens = [t for t in tokens if t not in self.stopwords]
                
            # Generate n-grams
            text_ngrams = list(ngrams(tokens, n))
            # Update counts
            ngram_counts.update(text_ngrams)
        
        # Convert to DataFrame
        ngram_df = pd.DataFrame(
            [(' '.join(k), v) for k, v in ngram_counts.items()],
            columns=['word', 'count']
        )
        
        # Sort by frequency
        ngram_df = ngram_df.sort_values('count', ascending=False).reset_index(drop=True)
        
        logger.info(f"Analyzed {n}-grams: found {len(ngram_df)} unique patterns")
        return ngram_df


    def run_all_analyses(self, save_figures: bool = True, 
                       keep_stopwords: bool = False) -> Dict[str, Any]:
        """
        Run all text analyses and generate visualizations.
        
        Args:
            save_figures: Whether to save generated figures
            keep_stopwords: Whether to keep stop words (default: remove stop words)
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        stopwords_status = "with" if keep_stopwords else "without"
        logger.info(f"Running text analyses {stopwords_status} stop words")
        
        # Word frequency analysis
        all_words_df, positive_words_df, negative_words_df = self.analyze_all_word_frequencies(
            keep_stopwords=keep_stopwords
        )
        
        results['word_frequencies'] = {
            'all': all_words_df,
            'positive': positive_words_df,
            'negative': negative_words_df
        }
        
        # Plot word frequencies
        title_suffix = " (with stop words)" if keep_stopwords else ""
        plot_top_words(
            all_words_df, n=50, 
            title=f"Most Frequent Words{title_suffix}", 
            save=save_figures
        )
        plot_top_words(
            positive_words_df, n=50, 
            title=f"Most Frequent Words (Positive){title_suffix}", 
            save=save_figures, fig_name="top_words_positive"
        )
        plot_top_words(
            negative_words_df, n=50, 
            title=f"Most Frequent Words (Negative){title_suffix}", 
            save=save_figures, fig_name="top_words_negative"
        )
        
        # Word comparison
        word_comparison = plot_sentiment_word_comparison(
            positive_words_df, negative_words_df, top_n=30, save=save_figures
        )
        results['word_comparison'] = word_comparison
        
        # Generate word clouds
        generate_wordcloud(
            all_words_df, 
            title=f"All Tweets{title_suffix}", 
            save=save_figures
        )
        generate_wordcloud(
            positive_words_df, 
            title=f"Positive Tweets{title_suffix}", 
            save=save_figures, fig_name="wordcloud_positive"
        )
        generate_wordcloud(
            negative_words_df, 
            title=f"Negative Tweets{title_suffix}", 
            save=save_figures, fig_name="wordcloud_negative"
        )
        
        # N-grams analysis
        bigram_df = self.analyze_ngrams(n=2, top_k=20, keep_stopwords=keep_stopwords)
        trigram_df = self.analyze_ngrams(n=3, top_k=20, keep_stopwords=keep_stopwords)
        results['ngrams'] = {
            'bigrams': bigram_df,
            'trigrams': trigram_df
        }
        
        plot_top_words(bigram_df, n=20, title="Most Frequent Bigrams", save=save_figures, fig_name="top_bigrams")
        plot_top_words(trigram_df, n=20, title="Most Frequent Trigrams", save=save_figures, fig_name="top_trigrams")
        
        # Hashtag analysis
        all_hashtags_df = self.extract_hashtags()
        positive_hashtags_df = self.extract_hashtags(self.positive_df)
        negative_hashtags_df = self.extract_hashtags(self.negative_df)
        results['hashtags'] = {
            'all': all_hashtags_df,
            'positive': positive_hashtags_df,
            'negative': negative_hashtags_df
        }
        
        # Plot hashtags
        plot_top_words(all_hashtags_df, n=50, title="Most Frequent Hashtags", 
                    save=save_figures, fig_name="top_hashtags")
        hashtag_comparison = plot_sentiment_word_comparison(
            positive_hashtags_df, negative_hashtags_df, top_n=30, save=save_figures
        )
        results['hashtag_comparison'] = hashtag_comparison
        
        # Mention analysis
        all_mentions_df = self.extract_mentions()
        positive_mentions_df = self.extract_mentions(self.positive_df)
        negative_mentions_df = self.extract_mentions(self.negative_df)
        results['mentions'] = {
            'all': all_mentions_df,
            'positive': positive_mentions_df,
            'negative': negative_mentions_df
        }
        
        # Plot mentions
        plot_top_words(all_mentions_df, n=50, title="Most Frequent Mentions", 
                    save=save_figures, fig_name="top_mentions")
        mention_comparison = plot_sentiment_word_comparison(
            positive_mentions_df, negative_mentions_df, top_n=30, save=save_figures
        )
        results['mention_comparison'] = mention_comparison
        
        
        
        # Punctuation analysis
        pos_punct_df, neg_punct_df = self.analyze_punctuation()
        results['punctuation'] = {
            'positive': pos_punct_df,
            'negative': neg_punct_df
        }
        self.plot_punctuation_comparison(pos_punct_df, neg_punct_df, save=save_figures)
        
        # URL analysis
        url_stats = self.analyze_urls()
        results['urls'] = url_stats
        
        # Capitalization analysis
        caps_stats = self.analyze_capitalization()
        results['capitalization'] = caps_stats
        
        logger.info("Completed all text analyses")
        return results