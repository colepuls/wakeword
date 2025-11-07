from dotenv import load_dotenv; load_dotenv()
import soundfile as sf, os, glob, torch, random, torchaudio
import torch.nn.functional as F

"""
This program gets data from datapaths, then:
- Splits data to 80/20.
- Combines all train data together and all val data together.
- Adds labels.
- Combines data with labels.
- Pads data.
- Converts log-mel spectrogram (prep for RNN).
- Converts data to features (Ready for RNN feed).
"""

# Get data paths
WAKEWORD_DATA_PATH = os.getenv("WAKEWORD_DATA_PATH")
wakeword_paths = glob.glob(f"{WAKEWORD_DATA_PATH}*.wav")

BACKGROUND_DATA_PATH = os.getenv("BACKGROUND_DATA_PATH")
background_paths = glob.glob(f"{BACKGROUND_DATA_PATH}*.wav")

# Store wakeword data
wakeword_data = []
for path in wakeword_paths:
    waveforms, sample_rate = sf.read(path)
    wakeword_data.append(torch.tensor(waveforms, dtype=torch.float32))

# Store background data
background_data = []
for path in background_paths:
    waveforms, sample_rate = sf.read(path)
    background_data.append(torch.tensor(waveforms, dtype=torch.float32))

# Split data
split_ww = int(0.8 * len(wakeword_data))
train_ww_data = wakeword_data[:split_ww]
val_ww_data = wakeword_data[split_ww:]
random.shuffle(train_ww_data); random.shuffle(val_ww_data)

split_bg = int(0.8 * len(background_data))
train_bg_data = background_data[:split_bg]
val_bg_data = background_data[split_bg:]

# Balance
train_data = train_ww_data + train_bg_data
val_data = val_ww_data + val_bg_data

# Label
train_labels = [1]*len(train_ww_data) + [0]*len(train_bg_data)
val_labels = [1]*len(val_ww_data) + [0]*len(val_bg_data)

# Combine and shuffle
train = list(zip(train_data, train_labels))
val = list(zip(val_data, val_labels))
random.shuffle(train); random.shuffle(val)

# Pad data
TARGET_LEN = 16000

pad_train = []
for x, y in train:
    if x.numel() < TARGET_LEN:
        x = F.pad(x, (0, TARGET_LEN - x.numel()))
    pad_train.append((x, y))
train = pad_train

pad_val = []
for x, y in val:
    if x.numel() < TARGET_LEN:
        x = F.pad(x, (0, TARGET_LEN - x.numel()))
    pad_val.append((x, y))
val = pad_val

# Convert audio to log-mel spectrogram for feeding into RNN
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=40)
db_transform = torchaudio.transforms.AmplitudeToDB()

def get_features(dataset):
    feats = []
    for x, y in dataset:
        mel = mel_transform(x)
        mel_db = db_transform(mel)
        feats.append((mel_db, y))
    return feats

train_feats = get_features(train)
val_feats = get_features(val)

if __name__ == '__main__':
    print(f"Sample rate: {sample_rate}\n")

    print(f"Lenght of wakeword data: {len(wakeword_data)}\n")
    print(f"Example idx of wakeword data: {wakeword_data[0]}\n")

    print(f"Length of background data: {len(background_data)}\n")
    print(f"Example idx of background data: {background_data[0]}\n")

    print(f"Length of train wakeword data: {len(train_ww_data)}\n")
    print(f"Length of val wakeword data: {len(val_ww_data)}\n")

    print(f"Length of train background data: {len(train_bg_data)}\n")
    print(f"Length of val background data: {len(val_bg_data)}\n")

    print(f"Length of TRAIN data: {len(train_data)}\n")
    print(f"Length of VAL data: {len(val_data)}\n")

    for i, (x, y) in enumerate(train[:50]):
        print(f"train[{i}] shape: {x.shape}, label: {y}\n")
        
    print(f"Min len: {min(x.numel() for x, _ in train)}, Max len: {max(x.numel() for x, _ in train)}\n")

    print(train_feats[0][0].shape, train_feats[0][1])