import os
import numpy as np
import pandas as pd
import librosa

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw", "audio")
META_FILE = os.path.join(BASE_DIR, "data", "raw", "meta", "esc50.csv")
FEATURE_DIR = os.path.join(BASE_DIR, "data", "processed", "features")

# Ensure output directory exists
os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_features(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """Load audio and compute spectrogram + MFCCs."""
    y, sr = librosa.load(file_path, sr=None)

    # Spectrogram (dB)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    return S_db, mfccs

def process_dataset():
    """Loop through all clips and save features."""
    meta = pd.read_csv(META_FILE)

    all_specs = []
    all_mfccs = []
    labels = []

    for idx, row in meta.iterrows():
        filename = row["filename"]
        category = row["category"]
        file_path = os.path.join(AUDIO_DIR, filename)

        try:
            S_db, mfccs = extract_features(file_path)

            # Save individual features as .npy
            prefix = os.path.splitext(filename)[0]
            spec_path = os.path.join(FEATURE_DIR, f"{prefix}_spec.npy")
            mfcc_path = os.path.join(FEATURE_DIR, f"{prefix}_mfcc.npy")

            np.save(spec_path, S_db)
            np.save(mfcc_path, mfccs)

            # Collect for dataset arrays
            all_specs.append(S_db)
            all_mfccs.append(mfccs)
            labels.append(category)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save combined arrays + labels
    np.save(os.path.join(FEATURE_DIR, "all_specs.npy"), np.array(all_specs, dtype=object))
    np.save(os.path.join(FEATURE_DIR, "all_mfccs.npy"), np.array(all_mfccs, dtype=object))
    np.save(os.path.join(FEATURE_DIR, "labels.npy"), np.array(labels))

    print(f"✅ Feature extraction complete. Saved to {FEATURE_DIR}")

if __name__ == "__main__":
    process_dataset()
