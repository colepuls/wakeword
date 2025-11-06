from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv
import soundfile as sf
import torch
import random
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

def convert_data(valww, valbg, trainww, trainbg):
    # Convert each dataset and add labels, wakeword = 1, background = 0
    valww = [(torch.tensor(w, dtype=torch.float32), 1) for (w, sr) in valww]
    valbg = [(torch.tensor(w, dtype=torch.float32), 0) for (w, sr) in valbg]
    trainww = [(torch.tensor(w, dtype=torch.float32), 1) for (w, sr) in trainww]
    trainbg = [(torch.tensor(w, dtype=torch.float32), 0) for (w, sr) in trainbg]

    # Combine datasets
    val_data = valww + valbg
    train_data = trainww + trainbg

    # shuffle data
    random.shuffle(val_data)
    random.shuffle(train_data)

    return val_data, train_data

class WakewordDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        waveform, label = self.data[idx]
        return waveform, torch.tensor(label, dtype=torch.long)

if __name__ == '__main__':
    val_ww, val_bg, train_ww, train_bg = split_data()

    val_data, train_data = convert_data(val_ww, val_bg, train_ww, train_bg)

    train_dataset = WakewordDataset(train_data)
    val_dataset = WakewordDataset(val_data)

    collate = lambda b: (
        pad_sequence([x for x, _ in b], batch_first=True),
        torch.tensor([y for _, y in b], dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate)

    for targets, labels in train_loader:
        print(targets)
        print(labels)
        break
