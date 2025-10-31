"""
- Do a 80/20 split on raw_data for both wakeword data and background data.
- 80 train, 20 val.
- train wakeword data into cleaned_data/train/wakeword
- train background data into cleaned_data/train/background
- val wakeword data into cleaned_data/val/wakeword
- val background data into cleaned_data/val/background
"""

from dotenv import load_dotenv
import soundfile as sf
import glob
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

VAL_SPLIT = 0.2

def load_data(path) -> list:
    files = glob.glob(f"{path}*.wav")
    data = []

    for f in files:
        waveform, sr = sf.read(f)
        data.append((waveform, sr))

    return data

def split_data():
    # Get raw data
    raw_wakeword_data = load_data(WAKEWORD_DATA_PATH)
    raw_background_data = load_data(OUTSIDE_DATA_PATH)

    # Get val data
    val_wakeword_split = int(len(raw_wakeword_data) * VAL_SPLIT)
    val_wakeword_data = raw_wakeword_data[:val_wakeword_split]

    val_background_split = int(len(raw_background_data) * VAL_SPLIT)
    val_background_data = raw_background_data[:val_background_split]

    # Get train data
    train_wakeword_data = raw_wakeword_data[val_wakeword_split:]
    train_background_data = raw_background_data[val_background_split:]

    # Print val lengths
    print(f"Val wakeword data length: {len(val_wakeword_data)}\n")
    print(f"Val background data length: {len(val_background_data)}\n")

    # Print train lengths
    print(f"Train wakeword data length: {len(train_wakeword_data)}\n")
    print(f"Train background data length: {len(train_background_data)}\n")
    

if __name__ == '__main__':
    split_data()