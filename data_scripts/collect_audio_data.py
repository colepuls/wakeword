from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import sys
import os

load_dotenv()

WAKEWORD_DATA_PATH = os.getenv("WAKEWORD_DATA_PATH")
OUTSIDE_DATA_PATH = os.getenv("OUTSIDE_DATA_PATH")

def record_wakeword_audio():
    samplerate = 16000
    block_size = 1024
    record_time = 1.0
    audio_number = 0

    while True:
        try:
            user_input = input("\nPress y to record 1 second of wakeword audio: ")
            print()
            if user_input == "y":
                start_time = time.time()
                audio = []
                with sd.InputStream(samplerate=samplerate, channels=1) as stream:
                    while True:
                        block, _ = stream.read(block_size)
                        audio.append(block)
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= record_time:
                            break
                    data = np.concatenate(audio, axis=0)
                    sf.write(f"{WAKEWORD_DATA_PATH}wakeword_audio_{audio_number}.wav", data, samplerate)
                    audio_number += 1
        except KeyboardInterrupt:
            print("\n\n\nProgram exited.\n")
            sys.exit()


def record_background_audio():
    samplerate = 16000
    block_size = 1024
    record_time = 1.0
    audio_number = 0

    print("Recording background audio...\n")
    while True:
        try:
            start_time = time.time()
            audio = []
            with sd.InputStream(samplerate=samplerate, channels=1) as stream:
                while True:
                    block, _ = stream.read(block_size)
                    audio.append(block)
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= record_time:
                        break
                data = np.concatenate(audio, axis=0)
                sf.write(f"{OUTSIDE_DATA_PATH}background_audio_{audio_number}.wav", data, samplerate)
                audio_number += 1
        except KeyboardInterrupt:
            print("\n\n\nProgram exited.\n")
            sys.exit()

if __name__ == '__main__':
    audio_to_collect = input("\nPress w to collect wakeword audio or b to collect background audio: ")
    print()
    if audio_to_collect == "w":
        record_wakeword_audio()
    elif audio_to_collect == "b":
        record_background_audio()
    else:
        print("Invalid input\n")