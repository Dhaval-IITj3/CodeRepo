import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Assume your data is organized in a directory where each subfolder is a bike brand/class,
# and contains the .ogg files for that brand. Replace with your actual path.
DATA_DIR_NAME = 'Resources\\EngineSoundsNewBikes'
BASE_DIR = Path(__file__).parent
BASE_DIR = Path(BASE_DIR, "../").resolve()
PLOT_DIR = Path.joinpath(BASE_DIR, 'EngineSoundAnalysis', 'Plots', 'NewBikeFeatureExtraction')
data_dir = str(Path.joinpath(BASE_DIR, DATA_DIR_NAME))

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

if not os.path.exists(data_dir):
    print(f"Error: {data_dir} does not exist.")
    exit()


# Function to extract features from an audio file
def extract_features(file_path):
    # Load the audio file (librosa supports .ogg format)
    y, sr = librosa.load(file_path, sr=22050)  # Resample to 22050 Hz for consistency

    # Extract MFCC (13 coefficients, mean over time)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # Extract Zero Crossing Rate (mean)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Extract Spectral Centroid (mean)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Extract Spectral Rolloff (mean)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Extract Chroma STFT (12 features, mean over time)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    # Concatenate all features into a single vector
    features = np.hstack([mfcc, [zcr], [centroid], [rolloff], chroma])
    return features


# Load data and extract features
classes = os.listdir(data_dir)
features_list = []
labels = []

for cls in classes:
    path = os.path.join(data_dir, cls)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith('.ogg'):
                path = os.path.join(path, file)
                try:
                    feat = extract_features(path)
                    features_list.append(feat)
                    labels.append(cls)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
    elif os.path.isfile(path) and path.endswith('.ogg'):
        try:
            feat = extract_features(path)
            features_list.append(feat)
            labels.append(cls)
        except Exception as e:
            print(f"Error processing {path}: {e}")


# Convert to numpy array and encode labels
if len(features_list) == 0:
    raise ValueError("No audio files found or processed.")
X = np.array(features_list)
le = LabelEncoder()
y = le.fit_transform(labels)

# Create a DataFrame for easier plotting and analysis
feature_names = [f'mfcc_{i + 1}' for i in range(13)] + ['zcr', 'centroid', 'rolloff'] + [f'chroma_{i + 1}' for i in
                                                                                         range(12)]
df = pd.DataFrame(X, columns=feature_names)
df['label'] = labels

# Plot scalar features (one plot per feature, boxplot for distributions across classes)
scalar_features = ['zcr', 'centroid', 'rolloff']
for feat in scalar_features:
    fig, ax = plt.subplots(figsize=(10, 12))  # Explicit figure and axis
    df.boxplot(column=feat, by='label', grid=False, ax=ax)
    ax.set_title(f'Distribution of {feat} by Bike Brand')
    ax.set_xlabel('Bike Brand')
    ax.set_ylabel(feat)
    plt.suptitle('')  # Remove automatic pandas title
    plt.xticks(rotation=45)
    plt.show()
    fig.savefig(os.path.join(f'{PLOT_DIR}', f'{feat}.png'), bbox_inches='tight')
    plt.close(fig)

# Plot mean MFCC coefficients for each class
plt.figure(figsize=(12, 6))
for cls in classes:
    class_df = df[df['label'] == cls]
    mean_mfcc = class_df[[f'mfcc_{i + 1}' for i in range(13)]].mean().values
    plt.plot(range(1, 14), mean_mfcc, label=cls)
plt.title('Mean MFCC Coefficients by Bike Brand')
plt.xlabel('MFCC Coefficient')
plt.ylabel('Value')
plt.legend()
plt.show()
plt.savefig(os.path.join(f'{PLOT_DIR}', 'MeanMFCCCoefficients.png'))
plt.close()

# Plot mean Chroma features for each class
plt.figure(figsize=(12, 6))
for cls in classes:
    class_df = df[df['label'] == cls]
    mean_chroma = class_df[[f'chroma_{i + 1}' for i in range(12)]].mean().values
    plt.plot(range(1, 13), mean_chroma, label=cls)
plt.title('Mean Chroma Features by Bike Brand')
plt.xlabel('Chroma Bin')
plt.ylabel('Value')
plt.legend()
plt.show()
plt.savefig(os.path.join(f'{PLOT_DIR}', 'MeanChromaFeatures.png'))
plt.close()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# To use the model for prediction on a new file:
# new_features = extract_features('path/to/new_sound.ogg')
# pred_label = le.inverse_transform(model.predict([new_features]))[0]
# print(f"Predicted Bike Brand: {pred_label}")