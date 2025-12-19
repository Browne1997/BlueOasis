import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import AudioCNN
from load_features import load_features


def load_real_mfcc():
    """
    Loads a single real MFCC from the processed dataset.
    Returns a tensor shaped (1, 1, n_mfcc, time_frames).
    """
    X, y = load_features(feature_type="mfcc")  # X is a list of MFCC arrays

    mfcc = X[0]  # shape: (n_mfcc, time_frames)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)

    # Add batch + channel dimensions → (1, 1, n_mfcc, time_frames)
    mfcc_tensor = mfcc_tensor.unsqueeze(0).unsqueeze(0)

    return mfcc_tensor, y[0]


def train_step(model, input_tensor, label, criterion, optimizer):
    """
    Runs a single forward/backward optimisation step.
    """
    optimizer.zero_grad()
    outputs = model(input_tensor)
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()
    return outputs, loss.item()


if __name__ == "__main__":
    num_classes = 50  # ESC-50 has 50 classes

    # Instantiate model
    model = AudioCNN(num_classes=num_classes)

    # Loss + optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n=== Loading one real MFCC ===")
    real_mfcc, real_label = load_real_mfcc()

    print("Real MFCC shape:", real_mfcc.shape)  # (1, 1, 13, time_frames)

    # Convert label to tensor
    real_label_tensor = torch.tensor([real_label_to_index := 0])  # placeholder label index

    # Forward + backward pass on real MFCC
    outputs, loss = train_step(
        model, real_mfcc, real_label_tensor, criterion, optimizer
    )

    print("Output shape (real MFCC):", outputs.shape)
    print("Loss (real MFCC):", loss)

    print("\n=== Running dummy batch example ===")
    # Dummy MFCC batch
    n_mfcc = 13
    time_frames = real_mfcc.shape[-1]  # match real MFCC length
    dummy_input = torch.randn(8, 1, n_mfcc, time_frames)
    dummy_labels = torch.randint(0, num_classes, (8,))

    outputs, loss = train_step(
        model, dummy_input, dummy_labels, criterion, optimizer
    )

    print("Output shape (dummy batch):", outputs.shape)
    print("Loss (dummy batch):", loss)

    print("\nDummy training script completed successfully.")

