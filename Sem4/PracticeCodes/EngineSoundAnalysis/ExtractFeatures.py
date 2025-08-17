import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path

# Static final variables
EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR_NAME = 'Resources'
BASE_DIR = Path(__file__).parent
BASE_DIR = Path(BASE_DIR, "../").resolve()
DATA_DIR = Path.joinpath(BASE_DIR, DATA_DIR_NAME)

BEFORE_DIR = Path(DATA_DIR, 'Before')
AFTER_DIR = Path(DATA_DIR, 'After')

before_list = os.listdir(BEFORE_DIR)
after_list = os.listdir(AFTER_DIR)

# Extract engine names from file names
engines = []

for f in before_list:
    if f.endswith('.ogg'):
        engines.append(f.split('_')[0])

for f in after_list:
    if f.endswith('.ogg'):
        engines.append(f.split('_')[0])

engines = list(set(engines))


# Step 1: Feature Extraction Function
def extract_features(file_path):
    waveform, sr = torchaudio.load(file_path)  # Loads OGG directly
    if waveform.shape[0] > 1:  # Convert to mono if stereo
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=13)
    mfcc = mfcc_transform(waveform)  # Shape: [1, 13, time_frames]
    mfcc_mean = torch.mean(mfcc, dim=2).squeeze().numpy()  # Average over time: [13]
    return mfcc_mean


# Step 2: Load Data (assumes folders like 'engine1_before/', etc.)
features = []
labels = []  # 0: before, 1: after

for engine in engines:
    for state in ['Before', 'After']:
        file = f'{engine}_{state}.ogg'
        file_path = os.path.join(DATA_DIR, state, file)

        if not os.path.exists(file_path):
            print(f'File not found: {file_path}')
            continue

        print(f'Extracting features from {file_path}')

        if file.endswith('.ogg'):
            feat = extract_features(file_path)
            features.append(feat)
            labels.append(0 if state == 'Before' else 1)

if not features:
    raise ValueError("No OGG files found. Check your folder structure.")

X = np.array(features)  # Shape: [num_samples, 13]
y = np.array(labels)  # Shape: [num_samples]

# Optional: Visualize feature differences (e.g., MFCC 0: energy)
before_feats = X[y == 0]
after_feats = X[y == 1]
plt.boxplot([before_feats[:, 0], after_feats[:, 0]], labels=['Before', 'After'])
plt.title('MFCC[0] Distribution Before vs. After Oil Change')
plt.show()

# Step 3: Prepare Data for PyTorch
# Simple train/test split (80/20); for small data, use cross-validation in practice
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().unsqueeze(1))
test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float().unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


# Step 4: Define Simple ML Model (Logistic Regression)
class SimpleClassifier(nn.Module):
    def __init__(self, input_size=13):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = SimpleClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Step 5: Train the Model
epochs = 50  # Adjust based on data size
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Step 6: Evaluate
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        predicted = (outputs > 0.5).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Step 7: Identify Distinguishing Features
# Weights: Higher abs value = more important for distinction
weights = model.linear.weight.data.squeeze().abs().numpy()
feature_importance = sorted(enumerate(weights), key=lambda x: x[1], reverse=True)
print("Top Distinguishing MFCC Features (by weight magnitude):")
for idx, imp in feature_importance[:5]:  # Top 5
    print(f'MFCC[{idx}]: Importance {imp:.4f}')

# To predict on a new file:
# new_feat = extract_features('new_recording.ogg')
# new_tensor = torch.from_numpy(new_feat).float().unsqueeze(0)
# prediction = model(new_tensor) > 0.5
# print('Predicted: After Oil Change' if prediction else 'Predicted: Before Oil Change')