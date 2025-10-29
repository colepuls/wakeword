import random
import shutil
import glob
from pathlib import Path

SOURCE = Path("/Users/colepuls/Desktop/projects/wakeword/data_raw")
DESTINATION = Path("/Users/colepuls/Desktop/projects/wakeword/data")
VAL_SPLIT = 0.2
ALLOWED_FILES = {".wav"}

random.seed(42)

def split_data(data_name: str):
    source_data = SOURCE / data_name

    wav_files = [p for p in source_data.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_FILES]

    if not wav_files:
        print("No wav files found.\n")
        return
    
    random.shuffle(wav_files)
    total = len(wav_files)
    val = int(round(total * VAL_SPLIT))
    val_set = set(wav_files[:val])

    for split in ["train", "val"]:
        (DESTINATION / split / data_name).mkdir(parents=True, exist_ok=True)

    for w in wav_files:
        split = "val" if w in val_set else "train"
        path = DESTINATION / split / data_name / w.name
        shutil.move(str(w), str(path))

    print(f"{data_name}: total={total}, train={total - val}, val={val}\n")

if __name__ == '__main__':
    for data in ["wake", "background"]:
        split_data(data)
