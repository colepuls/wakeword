from dotenv import load_dotenv; load_dotenv()
import soundfile as sf, os, glob, torch, random

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

split_bg = int(0.8 * len(background_data))
train_bg_data = background_data[:split_bg]
val_bg_data = background_data[split_bg:]

train_data = train_ww_data + train_bg_data
val_data = val_ww_data + val_bg_data
random.shuffle(train_data); random.shuffle(val_data)

if __name__ == '__main__':
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