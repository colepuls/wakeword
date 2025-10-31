"""
- Do a 80/20 split on raw_data for both wakeword data and background data.
- 80 train, 20 val.
- train wakeword data into cleaned_data/train/wakeword
- train background data into cleaned_data/train/background
- val wakeword data into cleaned_data/val/wakeword
- val background data into cleaned_data/val/background
"""

from dotenv import load_dotenv
import os

load_dotenv()

# Raw data paths
WAKEWORD_DATA_PATH = os.getenv("WAKEWORD_DATA_PATH")
OUTSIDE_DATA_PATH = os.getenv("OUTSIDE_DATA_PATH")

# Train data paths
WAKEWORD_TRAIN_PATH = os.getenv("WAKEWORD_TRAIN_PATH")
BACKGROUND_TRAIN_PATH = os.getenv("BACKGROUND_TRAIN_PATH")

# Val data paths
WAKEWORD_VAL_PATH = os.getenv("WAKEWORD_VAL_PATH")
BACKGROUND_VAL_PATH = os.getenv("BACKGROUND_VAL_PATH")

# def split_data():
    

