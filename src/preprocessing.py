import os
import pandas as pd
import librosa
import numpy as np

# Paths
AUDIO_DIR = "data/raw/audio"
META_FILE = "data/raw/meta/esc50.csv"

def load_metadata():
    """Load ESC-50 metadata CSV."""
    meta = pd.read_csv(META_FILE)
    return meta

def explore_metadata(meta):
    """Print basic dataset stats."""
    print("Number of clips:", len(meta))
    print("Number of classes:", meta['category'].nunique())
    print("Classes:", meta['category'].unique())
    print("\nClass distribution:")
    print(meta['category'].value_counts())
    print("\nClip durations (should all be ~5s):")
    durations = []
    for _, row in meta.sample(5).iterrows():  # sample a few
        file_path = os.path.join(AUDIO_DIR, row['filename'])
        y, sr = librosa.load(file_path, sr=None)
        durations.append(len(y)/sr)
    print(durations)

def load_audio(file_name, sr=None):
    """Load a single audio file."""
    file_path = os.path.join(AUDIO_DIR, file_name)
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def extract_spectrogram(y, sr):
    """Compute spectrogram in dB scale."""
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db

def extract_mfcc(y, sr, n_mfcc=13):
    """Compute MFCCs."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

if __name__ == "__main__":
    # Load metadata
    meta = load_metadata()
    explore_metadata(meta)

    # Example: load one file and extract features
    example_file = meta.iloc[0]['filename']
    y, sr = load_audio(example_file)
    print(f"Loaded {example_file} with sample rate {sr}, length {len(y)/sr:.2f}s")

    S_db = extract_spectrogram(y, sr)
    mfccs = extract_mfcc(y, sr)

    print("Spectrogram shape:", S_db.shape)
    print("MFCC shape:", mfccs.shape)

