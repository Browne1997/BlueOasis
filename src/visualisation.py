import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import pandas as pd
import os
import numpy as np

RAW_AUDIO_DIR = "data/raw/audio"
PROC_AUDIO_DIR = "data/processed/audio"
FEATURE_DIR = "data/processed/features"
META_FILE = "data/raw/meta/esc50.csv"

# -----------------------------
# Loading Helpers
# -----------------------------

def load_raw_audio(filename):
    path = os.path.join(RAW_AUDIO_DIR, filename)
    y, sr = librosa.load(path, sr=None)
    return y, sr

def load_processed_audio(filename):
    path = os.path.join(PROC_AUDIO_DIR, filename)
    y, sr = librosa.load(path, sr=None)
    return y, sr

def load_mfcc_feature(filename):
    """Load MFCC .npy file saved during feature extraction."""
    path = os.path.join(FEATURE_DIR, filename.replace(".wav", "_mfcc.npy"))
    if os.path.exists(path):
        return np.load(path)
    return None

def extract_spectrogram(y, sr, n_fft=2048, hop_length=512):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db

def extract_mfcc(y, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfccs


# -----------------------------
# Plotting Functions
# -----------------------------

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

def plot_mfcc(mfccs, title="Heatmap of MFCCs"):
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(mfccs, x_axis="time", ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_xlabel("time (frames)")
    ax.set_ylabel("MFCC coefficients (1–13)")
    ax.set(title=title)
    return fig

def plot_class_distribution():
    meta = pd.read_csv(META_FILE)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(y="category", data=meta, order=meta['category'].value_counts().index, ax=ax)
    ax.set(title="Class Distribution", xlabel="Count", ylabel="Category")
    return fig

def plot_duration_distribution(sample_size=200):
    meta = pd.read_csv(META_FILE)
    durations = []
    for _, row in meta.sample(sample_size).iterrows():
        file_path = os.path.join(RAW_AUDIO_DIR, row["filename"])
        y, sr = librosa.load(file_path, sr=None)
        durations.append(len(y)/sr)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(durations, bins=20, ax=ax)
    ax.set(title="Clip Duration Distribution", xlabel="Duration (s)", ylabel="Count")
    return fig





