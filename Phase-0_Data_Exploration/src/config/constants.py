from typing import Dict, List
from pathlib import Path

# Data paths
ROOT_DIR: Path = Path(__file__).parent.parent.parent.parent
PHASE_DIR: Path = ROOT_DIR / "Phase-0_Data_Exploration"
DATA_DIR: Path = ROOT_DIR / "Data"
REPORTS_DIR: Path = PHASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Ensure directories exist
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Dataset column names
TWEET_COLUMNS: List[str] = ["Target", "ID", "Date", "Query", "User", "Text"]

# Sentiment mapping
SENTIMENT_MAP: Dict[int, str] = {
    0: "Negative",
    4: "Positive",
    2: "Neutral"
}

# Regular expression patterns
PUNCT_PATTERNS: Dict[str, str] = {
    'period': r'\.',
    'comma': r',',
    'exclamation': r'!+',  # Catches multiple exclamation marks
    'question': r'\?+',    # Catches multiple question marks
    'semicolon': r';',
    'colon': r':',
    'quotes': r'["\']',
    'parentheses': r'[$$$$]',
    'brackets': r'[$$$$]',
    'braces': r'[\{\}]',
    'ellipsis': r'\.{3,}',  # Three or more periods
    'emoji_punctuation': r'[:;][\'"]?[-~]?[)\]DPp]'  # Basic emoji made with punctuation
}

WORD_PATTERN: str = r"(?:^|\s)([a-zA-Z'-]+)(?:[\s.,!?;]|$)"
HASHTAG_PATTERN: str = r'#\w+'
MENTION_PATTERN: str = r'@\w+'
URL_PATTERN: str = r'http[^ ]*'

# Time analysis
DAYS_OF_WEEK: List[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS: List[str] = ["January", "February", "March", "April", "May", "June", 
                     "July", "August", "September", "October", "November", "December"]