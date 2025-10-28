import sounddevice as sd
import soundfile as sf
import numpy as np
import time

def record_audio():
    samplerate = 16000
    block_size = 1024
    record_time = 1.0
    audio_number = 0

    while True:
        user_input = input("Press y to record 1 second of audio: ")
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
                sf.write(f"/Users/colepuls/Desktop/projects/picowakeword/data/pico_data/pico_audio_{audio_number}.wav", data, samplerate)
                audio_number += 1

if __name__ == '__main__':
    record_audio()