import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
META_FILE = os.path.join(BASE_DIR, "data", "raw", "meta", "esc50.csv")

# Fixed MFCC time dimension for ESC-50 (5 seconds @ hop_length=512)
FIXED_MFCC_LEN = 431  

def pad_or_truncate(mfcc, max_len=FIXED_MFCC_LEN):
    """
    Ensure MFCC has consistent time dimension by padding or truncating.
    mfcc shape: (13, time_frames)
    """
    current_len = mfcc.shape[1]

    if current_len < max_len:
        # Pad with zeros on the right
        pad_width = max_len - current_len
        return np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")

    elif current_len > max_len:
        # Truncate extra frames
        return mfcc[:, :max_len]

    return mfcc  # Already correct length


def load_features(feature_type="mfcc"):
    """
    Load per-clip features and labels into arrays.
    feature_type: "mfcc" or "spec"
    """
    meta = pd.read_csv(META_FILE)
    X, y = [], []

    for _, row in meta.iterrows():
        filename = row["filename"]
        category = row["category"]
        prefix = os.path.splitext(filename)[0]

        feature_file = os.path.join(FEATURE_DIR, f"{prefix}_{feature_type}.npy")

        if os.path.exists(feature_file):
            feat = np.load(feature_file, allow_pickle=True)

            # Apply padding/truncation ONLY for MFCCs
            if feature_type == "mfcc":
                feat = pad_or_truncate(feat, max_len=FIXED_MFCC_LEN)

            X.append(feat)
            y.append(category)

    return np.array(X), np.array(y)


def get_train_test(feature_type="mfcc", test_size=0.2, random_state=42):
    """
    Build train/test split from per-clip features.
    """
    X, y = load_features(feature_type=feature_type)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test(feature_type="mfcc")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Example feature shape: {X_train[0].shape}")
