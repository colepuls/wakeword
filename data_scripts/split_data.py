from dotenv import load_dotenv
import soundfile as sf
import torch, torchaudio
import glob
import os

load_dotenv()

# Raw data paths
WAKEWORD_DATA_PATH = os.getenv("WAKEWORD_DATA_PATH")
OUTSIDE_DATA_PATH = os.getenv("OUTSIDE_DATA_PATH")

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

    # Make splits
    val_wakeword_split = int(len(raw_wakeword_data) * VAL_SPLIT)
    val_background_split = int(len(raw_background_data) * VAL_SPLIT)

    # Get val data
    val_wakeword_data = raw_wakeword_data[:val_wakeword_split]
    val_background_data = raw_background_data[:val_background_split]

    # Get train data
    train_wakeword_data = raw_wakeword_data[val_wakeword_split:]
    train_background_data = raw_background_data[val_background_split:]

    return val_wakeword_data, val_background_data, train_wakeword_data, train_background_data

"""
Next step is to convert arrays into tensors for RNN using torchaudio.
"""

def convert_data():
    # Get val and train data
    val_ww, val_bg, train_ww, train_bg = split_data()

    # Convert each dataset
    val_ww = [(torch.tensor(w, dtype=torch.float32), sr) for (w, sr) in val_ww]
    val_bg = [(torch.tensor(w, dtype=torch.float32), sr) for (w, sr) in val_bg]
    train_ww = [(torch.tensor(w, dtype=torch.float32), sr) for (w, sr) in train_ww]
    train_bg = [(torch.tensor(w, dtype=torch.float32), sr) for (w, sr) in train_bg]

    return val_ww, val_bg, train_ww, train_bg




if __name__ == '__main__':
    val_ww, val_bg, train_ww, train_bg = convert_data()

    print(val_ww[0])
