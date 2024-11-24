import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Ensure GPU usage
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Clear cache
gc.collect()
if device.type == 'cuda':
    torch.cuda.empty_cache()

# Perform a simple computation to test GPU
a = torch.tensor([[1.0, 2.0]], device=device)
b = torch.tensor([[3.0], [4.0]], device=device)
c = torch.matmul(a, b)
print("Result of matrix multiplication:", c)

epochs = 4
bsize = 2048
# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bsize, shuffle=False)

# Define ResNet-15 model
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += shortcut
        x = self.relu(x)
        return x

class ResNet15(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(ResNet15, self).__init__()
        self.initial = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Residual Block 1
        self.residual_block1 = nn.Sequential(
            *[ResidualBlock(64) for _ in range(3)]
        )

        # Residual Block 2
        self.transition = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.residual_block2 = nn.Sequential(
            *[ResidualBlock(128) for _ in range(3)]
        )

        # Final Layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.relu(x)
        x = self.residual_block1(x)
        x = self.transition(x)
        x = self.residual_block2(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Instantiate model, loss, and optimizer
model = ResNet15().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
import time

# Training loop with timing
for epoch in range(epochs):
    start_time = time.time()  # Start time of the epoch
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_time = time.time() - start_time  # Calculate time taken for the epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}, "
          f"Accuracy: {100. * correct / total:.2f}%, Time: {epoch_time:.2f} seconds")

# Validation loop
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")