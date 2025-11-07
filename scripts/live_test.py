import time
import threading
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F

from model import WakewordRNN
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

SAMPLE_RATE = 16000
TARGET_LEN = 16000
HOP_SECONDS = 0.25
COOLDOWN_SECONDS = 1.5
THRESH = 0.5
RMS_VAD = 1e-3

device = torch.device("cpu")
model = WakewordRNN().to(device)
model.load_state_dict(torch.load("wakeword_model.pth", map_location=device))
model.eval()

mel_transform = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    hop_length=160,
    n_mels=40
)
db_transform = AmplitudeToDB()

buf = np.zeros(TARGET_LEN, dtype=np.float32)
write_idx = 0
filled = False
lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    global write_idx, filled
    if status:
        pass
    mono = indata[:, 0].astype(np.float32)  # (frames,)
    with lock:
        n = mono.shape[0]
        end = write_idx + n
        if end <= TARGET_LEN:
            buf[write_idx:end] = mono
        else:
            first = TARGET_LEN - write_idx
            buf[write_idx:] = mono[:first]
            buf[:end % TARGET_LEN] = mono[first:]
        write_idx = (write_idx + n) % TARGET_LEN
        if not filled and write_idx == 0:
            filled = True

def get_current_window() -> torch.Tensor:
    with lock:
        if not filled and write_idx == 0:
            # not enough audio yet
            return None
        if filled:
            w = np.concatenate([buf[write_idx:], buf[:write_idx]])
        else:
            w = np.zeros(TARGET_LEN, dtype=np.float32)
            w[:write_idx] = buf[:write_idx]
    x = torch.from_numpy(w)
    if x.numel() < TARGET_LEN:
        x = F.pad(x, (0, TARGET_LEN - x.numel()))
    else:
        x = x[:TARGET_LEN]
    return x

def predict_prob(x_1d: torch.Tensor) -> float:
    # x_1d: shape [16000]
    mel = mel_transform(x_1d)
    mel_db = db_transform(mel).unsqueeze(0)  # (1, 40, T)
    with torch.no_grad():
        prob = model(mel_db.to(device)).item()
    return prob

def main():
    print("Listening...\n")

    last_check = 0.0
    last_detect = -1e9

    blocksize = int(SAMPLE_RATE * 0.10)  # 100 ms chunks to keep latency low

    try:
        with sd.InputStream(channels=1,
                            samplerate=SAMPLE_RATE,
                            dtype="float32",
                            blocksize=blocksize,
                            callback=audio_callback):
            while True:
                now = time.time()
                # check on a hop schedule
                if now - last_check >= HOP_SECONDS:
                    last_check = now
                    x = get_current_window()
                    if x is None:
                        continue

                    # simple VAD to skip silence
                    rms = float(torch.sqrt(torch.mean(x**2)).item())
                    if rms < RMS_VAD:
                        continue

                    prob = predict_prob(x)
                    if prob > THRESH and (now - last_detect) >= COOLDOWN_SECONDS:
                        last_detect = now
                        print(f"[{time.strftime('%H:%M:%S')}] Wakeword detected | prob={prob:.3f} | rms={rms:.4f}")

                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
