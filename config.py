import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# Constants
PRODUCT_CATEGORIES = ["electronics", "clothing", "home_garden", "sports", "beauty"]
FORECAST_HORIZON_DAYS = 7
ELASTICITY_MIN_SAMPLES = 30
COST_MARGIN_FLOOR = 0.15
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
LLM_MODEL = "claude-sonnet-4-6"

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
