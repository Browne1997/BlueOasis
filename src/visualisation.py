import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import pandas as pd
import os
import numpy as np

AUDIO_DIR = "data/raw/audio"
META_FILE = "data/raw/meta/esc50.csv"
PLOT_DIR = "data/processed/plots"

# Ensure output directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_waveform(y, sr, title="Waveform"):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title=title, xlabel="Time (s)", ylabel="Amplitude")
    return fig

def plot_spectrogram(S_db, sr, title="Spectrogram (dB)"):
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title=title)
    return fig

def plot_mfcc(mfccs, title="MFCCs"):
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(mfccs, x_axis="time", ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=title)
    return fig

def plot_class_distribution():
    """Plot histogram of class counts from metadata."""
    meta = pd.read_csv(META_FILE)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(y="category", data=meta, order=meta['category'].value_counts().index, ax=ax)
    ax.set(title="Class Distribution", xlabel="Count", ylabel="Category")
    return fig

def plot_duration_distribution(sample_size=200):
    """Plot histogram of clip durations (ESC-50 clips are ~5s)."""
    meta = pd.read_csv(META_FILE)
    durations = []
    for _, row in meta.sample(sample_size).iterrows():
        file_path = os.path.join(AUDIO_DIR, row["filename"])
        y, sr = librosa.load(file_path, sr=None)
        durations.append(len(y)/sr)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(durations, bins=20, ax=ax)
    ax.set(title="Clip Duration Distribution", xlabel="Duration (s)", ylabel="Count")
    return fig

# ---------------- DEMO BLOCK ----------------
if __name__ == "__main__":
    print("Running visualization demo...")

    # Load metadata
    meta = pd.read_csv(META_FILE)

    # Pick one example file
    example_file = meta.iloc[0]["filename"]
    file_path = os.path.join(AUDIO_DIR, example_file)
    y, sr = librosa.load(file_path, sr=None)

    # Compute features
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Build filename prefix based on clip name (without extension)
    prefix = os.path.splitext(example_file)[0]

    # Save plots to PLOT_DIR with descriptive names
    plot_waveform(y, sr, title=f"Waveform: {example_file}").savefig(os.path.join(PLOT_DIR, f"{prefix}_waveform.png"))
    plot_spectrogram(S_db, sr, title=f"Spectrogram: {example_file}").savefig(os.path.join(PLOT_DIR, f"{prefix}_spectrogram.png"))
    plot_mfcc(mfccs, title=f"MFCCs: {example_file}").savefig(os.path.join(PLOT_DIR, f"{prefix}_mfccs.png"))
    plot_class_distribution().savefig(os.path.join(PLOT_DIR, "class_distribution.png"))
    plot_duration_distribution().savefig(os.path.join(PLOT_DIR, "duration_distribution.png"))

    print(f"Plots saved to {PLOT_DIR}")




