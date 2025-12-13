import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
META_FILE = os.path.join(BASE_DIR, "data", "raw", "meta", "esc50.csv")

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
            X.append(feat)
            y.append(category)

    return np.array(X, dtype=object), np.array(y)

def get_train_test(feature_type="mfcc", test_size=0.2, random_state=42):
    """
    Build train/test split from per-clip features.
    """
    X, y = load_features(feature_type=feature_type)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test = get_train_test(feature_type="mfcc")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Example feature shape: {X_train[0].shape}")
