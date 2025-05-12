"""
Constants for the model training module.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

# Data paths
ROOT_DIR: Path = Path(__file__).parent.parent.parent.parent
PHASE_DIR: Path = ROOT_DIR / "Phase-1_Model_Training"
DATA_DIR: Path = ROOT_DIR / "Data"
REPORTS_DIR: Path = PHASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODEL_DIR: Path = PHASE_DIR / "models"
LOGS_DIR: Path = PHASE_DIR / "logs"
TENSORBOARD_DIR: Path = LOGS_DIR / "tensorboard"

# Data settings
RAW_DATA_URL: str = "raj713335/twittesentimentanalysis"
DATA_ENCODING: str = "ISO-8859-1"
COLUMN_NAMES: List[str] = ["Target", "ID", "Date", "Query", "User", "Text"]

# MLflow settings
MLFLOW_TRACKING_URI: str = "file:" + str(PHASE_DIR / "mlruns")
EXPERIMENT_NAME: str = "twitter_sentiment_analysis"

# Ensure directories exist
for directory in [REPORTS_DIR, FIGURES_DIR, MODEL_DIR, LOGS_DIR, TENSORBOARD_DIR]:
    directory.mkdir(exist_ok=True, parents=True)