# ML-Assignment - Supervised Regression

## Problem Description

Using Deep Learning techniques, predict the coordinates (x,y) of a pixel which has a value of 255 for 1 pixel in a given 50x50 pixel grayscale image and all other pixels are 0. The pixel with a value of 255 is randomly assigned. You may generate a dataset as required for solving the problem. Please explain your rationale behind dataset choices.

## Approach Overview

Dataset Generation

A synthetic dataset is generated since no real-world dataset is available.

Each image is generated programmatically with:

Size: 50 × 50

One pixel set to 255

Remaining pixels set to 0

Labels are normalized (x, y) coordinates of the bright pixel.

Model Selection

A Convolutional Neural Network (CNN) is used.

CNNs are well-suited for spatial feature extraction in images.

The network outputs two values corresponding to (x, y) coordinates.

Loss Function

Mean Squared Error (MSE) is used as this is a regression problem.

## Technologies Used:

Python 3.x

PyTorch

NumPy

Matplotlib

Colab Research Google

## Install Dependencies:
```py
!pip install numpy matplotlib torch torchvision
```


## Program:
```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
class PixelDataset(Dataset):
    """
    Dataset generating 50x50 images with one bright pixel.
    Label is (x, y) coordinate of the pixel.
    """

    def __init__(self, num_samples=5000):
        self.images = []
        self.labels = []

        for _ in range(num_samples):
            image = np.zeros((50, 50), dtype=np.float32)
            x = np.random.randint(0, 50)
            y = np.random.randint(0, 50)

            image[y, x] = 255.0

            self.images.append(image / 255.0)
            self.labels.append([x / 49.0, y / 49.0])  # normalized

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx]).unsqueeze(0)
        label = torch.tensor(self.labels[idx])
        return img, label
class PixelLocatorCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 12 * 12, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)
dataset = PixelDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PixelLocatorCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.show()
model.eval()

test_dataset = PixelDataset(num_samples=100)
test_loader = DataLoader(test_dataset, batch_size=100)

images, labels = next(iter(test_loader))
with torch.no_grad():
    predictions = model(images)

gt = labels.numpy() * 49
pred = predictions.numpy() * 49

plt.scatter(gt[:, 0], gt[:, 1], label="Ground Truth", alpha=0.6)
plt.scatter(pred[:, 0], pred[:, 1], label="Prediction", alpha=0.6)
plt.legend()
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Ground Truth vs Predicted Coordinates")
plt.grid()
plt.show()
```
## Output:
<img width="576" height="455" alt="download" src="https://github.com/user-attachments/assets/a259eb8b-a28f-449a-b1f9-7f7b72e4f141" />
<img width="562" height="455" alt="download" src="https://github.com/user-attachments/assets/d69ead39-fc4b-4e6d-a9a7-918d87efae05" />

