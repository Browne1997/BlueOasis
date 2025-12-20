import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, os

# Add project root to sys.path so imports work when running from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.model import AudioCNN
from src.load_features import load_features
from sklearn.preprocessing import LabelEncoder


def load_real_mfcc():
    """
    Loads a single real MFCC from the processed dataset.
    Returns:
        mfcc_tensor: shape (1, 1, n_mfcc, time_frames)
        label: original string label
    """
    X, y = load_features(feature_type="mfcc")  # X is numpy array of MFCCs

    mfcc = X[0]  # shape: (13, 431)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)

    # Add batch + channel dims → (1, 1, 13, 431)
    mfcc_tensor = mfcc_tensor.unsqueeze(0).unsqueeze(0)

    return mfcc_tensor, y[0]


def train_step(model, input_tensor, label_tensor, criterion, optimizer):
    """
    Runs a single forward/backward optimisation step.
    """
    optimizer.zero_grad()
    outputs = model(input_tensor)
    loss = criterion(outputs, label_tensor)
    loss.backward()
    optimizer.step()
    return outputs, loss.item()


if __name__ == "__main__":
    num_classes = 50  # ESC-50 has 50 classes
    input_shape = (1, 13, 431)

    # Instantiate model with correct input shape
    model = AudioCNN(input_shape=input_shape, num_classes=num_classes)

    # Loss + optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n=== Loading one real MFCC ===")
    real_mfcc, real_label = load_real_mfcc()
    print("Real MFCC shape:", real_mfcc.shape)  # (1, 1, 13, 431)

    # Encode label (ESC-50 labels are strings)
    le = LabelEncoder()
    le.fit([real_label])  # fit on single label just for dummy test
    real_label_tensor = torch.tensor([le.transform([real_label])[0]], dtype=torch.long)

    # Forward + backward pass on real MFCC
    outputs, loss = train_step(model, real_mfcc, real_label_tensor, criterion, optimizer)

    print("Output shape (real MFCC):", outputs.shape)
    print("Loss (real MFCC):", loss)

    print("\n=== Running dummy batch example ===")
    # Dummy MFCC batch
    n_mfcc = 13
    time_frames = real_mfcc.shape[-1]  # should be 431
    dummy_input = torch.randn(8, 1, n_mfcc, time_frames)
    dummy_labels = torch.randint(0, num_classes, (8,))

    outputs, loss = train_step(model, dummy_input, dummy_labels, criterion, optimizer)

    print("Output shape (dummy batch):", outputs.shape)
    print("Loss (dummy batch):", loss)

    print("\nDummy training script completed successfully.")


