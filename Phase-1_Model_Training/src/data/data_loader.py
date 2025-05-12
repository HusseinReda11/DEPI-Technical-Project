"""
Data loading and preparation utilities for model training.
"""
import pandas as pd
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

from ..config.constants import RAW_DATA_URL, DATA_ENCODING, COLUMN_NAMES, DATA_DIR
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class TweetDataset(Dataset):
    """Dataset class for tokenized tweet data."""
    
    def __init__(self, texts: List[str], targets: List[int], tokenizer: PreTrainedTokenizer, max_len: int = 48) -> None:
        """
        Initialize the dataset.
        
        Args:
            texts: List of tweet texts
            targets: List of sentiment labels (0 for negative, 1 for positive)
            tokenizer: Pretrained tokenizer
            max_len: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset item.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing input_ids, attention_mask, and targets
        """
        text = str(self.texts[idx])
        target = self.targets[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors=None
        )

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }

class DataManager:
    """Class for managing Twitter sentiment data."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the data manager.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.dataset_path: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
    def download_dataset(self) -> str:
        """
        Download the dataset from Kaggle.
        
        Returns:
            Path to the downloaded dataset
        """
        logger.info(f"Downloading dataset from {RAW_DATA_URL}")
        self.dataset_path = kagglehub.dataset_download(RAW_DATA_URL)
        logger.info(f"Dataset downloaded to {self.dataset_path}")
        return self.dataset_path
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the raw Twitter dataset.
        
        Returns:
            DataFrame containing raw tweet data
        """
        if self.dataset_path is None:
            self.download_dataset()
            
        logger.info("Loading raw Twitter data")
        file_path = f"{self.dataset_path}/tweets.csv"
        
        self.df = pd.read_csv(
            file_path, 
            encoding=DATA_ENCODING, 
            names=COLUMN_NAMES, 
            header=None
        )
        
        logger.info(f"Loaded {len(self.df)} tweets")
        return self.df
    
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for model training.
        
        Args:
            tokenizer: Pretrained tokenizer for text processing
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self.df is None:
            self.load_data()
            
        logger.info("Preparing data for training")
        
        # Map target values to binary labels
        self.df['Target'] = self.df['Target'].map({0: 0, 4: 1})
        
        # Sample data
        n_samples = len(self.df) // (2 * self.config['portion'])
        df_sampled = pd.concat([
            self.df[self.df['Target'] == 0].sample(n=n_samples, random_state=self.config['seed']),
            self.df[self.df['Target'] == 1].sample(n=n_samples, random_state=self.config['seed'])
        ]).sample(frac=1, random_state=self.config['seed'])
        
        logger.info(f"Using {len(df_sampled)} samples after sampling")
        
        # Split data
        train_df, val_df = train_test_split(
            df_sampled, 
            test_size=0.1, 
            random_state=self.config['seed'],
            stratify=df_sampled['Target']
        )
        
        logger.info(f"Split data: {len(train_df)} training samples, {len(val_df)} validation samples")
        
        # Create datasets
        train_dataset = TweetDataset(
            texts=train_df['Text'].values,
            targets=train_df['Target'].values,
            tokenizer=tokenizer,
            max_len=self.config['max_len']
        )
        
        val_dataset = TweetDataset(
            texts=val_df['Text'].values,
            targets=val_df['Target'].values,
            tokenizer=tokenizer,
            max_len=self.config['max_len']
        )
        
        # Create DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"Created DataLoaders with {len(self.train_loader)} training batches and {len(self.val_loader)} validation batches")
        
        return self.train_loader, self.val_loader