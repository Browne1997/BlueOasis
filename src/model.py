
# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(AudioCNN, self).__init__()
        # Input shape: (batch, 1, n_mfcc, time_frames)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * (n_mfcc//4) * (time_frames//4), 64)  # adjust dims
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # flatten all but batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
if __name__ == "__main__":
    # Dummy input: batch of 8 MFCCs with shape (batch, channels, n_mfcc, time_frames)
    n_mfcc, time_frames = 13, 100
    model = AudioCNN(num_classes=10)
    dummy_input = torch.randn(8, 1, n_mfcc, time_frames)
    output = model(dummy_input)
    print(output.shape)  # (8, 10)
