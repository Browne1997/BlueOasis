import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

RAW_DIR = "data/raw/audio"
OUT_DIR = "data/processed/audio"
META_FILE = "data/raw/meta/esc50.csv"

os.makedirs(OUT_DIR, exist_ok=True)

def preprocess_audio(file_name):
    in_path = os.path.join(RAW_DIR, file_name)
    out_path = os.path.join(OUT_DIR, file_name)

    y, sr = librosa.load(in_path, sr=44100, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Normalise amplitude
    y = y / np.max(np.abs(y))

    # Save processed audio
    sf.write(out_path, y, sr)

    return out_path

if __name__ == "__main__":
    meta = pd.read_csv(META_FILE)

    print("Processing audio files...")
    for file_name in meta["filename"]:
        preprocess_audio(file_name)

    print("Done! Processed audio saved to:", OUT_DIR)


