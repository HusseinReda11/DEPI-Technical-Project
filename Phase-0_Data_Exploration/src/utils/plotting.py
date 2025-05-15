"""
Plotting utilities for the Twitter sentiment analysis project.

This module contains helper functions for creating visualizations.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional
from wordcloud import WordCloud
from pathlib import Path
import sys

# Add the project root to the path so we can import modules correctly
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import (
    FIGURE_SIZE_MEDIUM, FIGURE_SIZE_LARGE, FIGURE_SIZE_SQUARE,
    FIGURE_DPI
)
from src.config.constants import FIGURES_DIR

def save_figure(fig_name: str) -> None:
    """
    Save the current matplotlib figure.
    
    Args:
        fig_name: Name to save the figure as (without extension)
    """
    plt.savefig(FIGURES_DIR / f"{fig_name}.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

def plot_sentiment_distribution(df: pd.DataFrame, target_col: str = 'Target', 
                               save: bool = True) -> None:
    """
    Plot a pie chart of sentiment distribution.
    
    Args:
        df: DataFrame containing the data
        target_col: Column name containing sentiment labels
        save: Whether to save the figure
    """
    plt.figure(figsize=FIGURE_SIZE_SQUARE)
    plt.pie(
        df[target_col].value_counts(), 
        labels=['negative', 'positive'], 
        autopct='%1.1f%%', 
        startangle=90,
        colors=['#e74c3c', '#2ecc71']
    )
    plt.title("Sentiment Distribution")
    plt.axis('equal')
    
    if save:
        save_figure("sentiment_distribution")
    else:
        plt.show()

def plot_top_words(freq_df: pd.DataFrame, n: int = 20, 
                  title: str = 'Top Most Frequent Words',
                  save: bool = True, fig_name: Optional[str] = None) -> None:
    """
    Plot bar chart of top n most frequent words.
    
    Args:
        freq_df: DataFrame with 'word' and 'count' columns
        n: Number of top words to plot
        title: Chart title
        save: Whether to save the figure
        fig_name: Name to save the figure as (defaults to title)
    """
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    sns.barplot(
        data=freq_df.head(n), 
        x='word', 
        y='count'
    )
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {n} {title}')
    plt.tight_layout()
    
    if save:
        fig_name = fig_name or f"top_{n}_{title.lower().replace(' ', '_')}"
        save_figure(fig_name)
    else:
        plt.show()

def plot_sentiment_word_comparison(positive_df: pd.DataFrame, negative_df: pd.DataFrame, 
                                 top_n: int = 20, save: bool = True) -> pd.DataFrame:
    """
    Plot a comparison of word frequencies between positive and negative sentiments.
    
    Args:
        positive_df: DataFrame with word frequencies in positive tweets
        negative_df: DataFrame with word frequencies in negative tweets
        top_n: Number of top words to compare
        save: Whether to save the figure
        
    Returns:
        DataFrame containing the comparison data
    """
    # Get the top N words from each sentiment
    top_pos = set(positive_df['word'].head(top_n))
    top_neg = set(negative_df['word'].head(top_n))
    
    # Find common words
    common_words = top_pos.intersection(top_neg)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'word': list(common_words),
        'Positive': [positive_df[positive_df['word'] == word]['count'].iloc[0] 
                    for word in common_words],
        'Negative': [negative_df[negative_df['word'] == word]['count'].iloc[0] 
                    for word in common_words]
    })
    
    # Sort by total frequency
    comparison_df['total'] = comparison_df['Positive'] + comparison_df['Negative']
    comparison_df = comparison_df.sort_values('total', ascending=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    
    # Plot horizontal bars
    y_pos = np.arange(len(comparison_df))
    width = 0.35
    
    # Create bars
    ax.barh(y_pos - width/2, comparison_df['Positive'], width, 
            label='Positive', color='#2ecc71', alpha=0.7)
    ax.barh(y_pos + width/2, comparison_df['Negative'], width,
            label='Negative', color='#e74c3c', alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Words')
    ax.set_title('Word Frequencies in Positive vs Negative Tweets')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['word'])
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels on the bars
    for i, v in enumerate(comparison_df['Positive']):
        ax.text(v, i - width/2, str(v), va='center', fontweight='bold')
    for i, v in enumerate(comparison_df['Negative']):
        ax.text(v, i + width/2, str(v), va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        save_figure("sentiment_word_comparison")
    else:
        plt.show()
    
    return comparison_df

def generate_wordcloud(word_freq_df: pd.DataFrame, title: str = "Word Cloud",
                      save: bool = True, fig_name: Optional[str] = None) -> None:
    """
    Generate and display a word cloud from word frequencies.
    
    Args:
        word_freq_df: DataFrame with 'word' and 'count' columns
        title: Title for the word cloud
        save: Whether to save the figure
        fig_name: Name to save the figure as (defaults to title)
    """
    wordcloud = WordCloud(
        width=1600, 
        height=800,
        background_color='white',
        max_words=300,
        max_font_size=200,
        min_font_size=10,
        random_state=1
    ).generate_from_frequencies(word_freq_df.set_index('word').to_dict()['count'])
    
    # Display
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    
    if save:
        fig_name = fig_name or f"wordcloud_{title.lower().replace(' ', '_')}"
        save_figure(fig_name)
    else:
        plt.show()