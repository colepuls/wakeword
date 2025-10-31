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

def load_data(path) -> list:
    files = glob.glob(f"{path}*.wav")
    data = []

    for f in files:
        waveform, sr = sf.read(f)
        data.append((waveform, sr))

    print(len(data))
    print(type(data))

    return data

# def split_data():

if __name__ == '__main__':
    load_data(WAKEWORD_DATA_PATH)
    load_data(OUTSIDE_DATA_PATH)

