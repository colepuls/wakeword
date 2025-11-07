from model import WakewordRNN
import torchaudio, torch, soundfile as sf
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# Load model
model = WakewordRNN()
model.load_state_dict(torch.load("wakeword_model.pth"))
model.eval()

SAMPLE_RATE = 16000
TARGET_LEN = 16000

mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=40)
db_transform = AmplitudeToDB()

path = "/Users/colepuls/Desktop/projects/wakeword/test_data/test2.wav"
waveform, sr = sf.read(path)
waveform = torch.tensor(waveform, dtype=torch.float32)

if waveform.ndim == 2:
    waveform = waveform.mean(dim=1)

if sr != SAMPLE_RATE:
    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

if waveform.numel() < TARGET_LEN:
    waveform = torch.nn.functional.pad(waveform, (0, TARGET_LEN - waveform.numel()))
else:
    waveform = waveform[:TARGET_LEN]

mel = mel_transform(waveform)
mel_db = db_transform(mel)
mel_db = mel_db.unsqueeze(0)  # add batch dim

# Inference
with torch.no_grad():
    pred = model(mel_db)
    prob = pred.item()

print(f"Prediction: {prob:.3f}")
if prob > 0.5:
    print("Wakeword detected")
else:
    print("Background / no wakeword")
