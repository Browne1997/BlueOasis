# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, input_shape=(1, 13, 431), num_classes=50):
        super(AudioCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Compute flattened dimension dynamically
        self.flatten_dim = self._get_flatten_dim(input_shape)

        # Fully connected layers
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _get_flatten_dim(self, input_shape):
        """
        Pass a dummy tensor through conv/pool layers to compute the flattened size.
        """
        x = torch.zeros(1, *input_shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Example usage
if __name__ == "__main__":
    model = AudioCNN(input_shape=(1, 13, 431), num_classes=10)
    dummy_input = torch.randn(8, 1, 13, 431)
    output = model(dummy_input)
    print(output.shape)
