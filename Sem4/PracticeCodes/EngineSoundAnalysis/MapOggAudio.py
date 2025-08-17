import pyogg
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def read_and_map_audio(input_file):
    # Open the OGG file using PyOgg
    ogg_file = pyogg.VorbisFile(input_file)

    # Get some basic info about the file (like sample rate, channels, etc.)
    sample_rate = ogg_file.info().rate
    channels = ogg_file.info().channels
    total_samples = ogg_file.info().length

    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Total Samples: {total_samples}")

    # Decode the audio data to get the raw sample data (numpy array)
    decoded_data = ogg_file.decode()

    # Convert to numpy array (it should already be in numpy format after decode)
    decoded_data = np.array(decoded_data)

    # Map the audio data - For example, plotting the waveform of the first channel
    if channels > 1:
        # If stereo, we select the first channel (left channel)
        decoded_data = decoded_data[:, 0]

    # Display the waveform of the first channel
    plt.figure(figsize=(10, 4))
    plt.plot(decoded_data[:10000])  # Plot only the first 10,000 samples to avoid large plots
    plt.title(f"Waveform of the first channel of {input_file}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

    return decoded_data

# Example Usage
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

# Step 2: Load Data (assumes folders like 'engine1_before/', etc.)
features = []

for engine in engines:
    for state in ['Before', 'After']:
        file = f'{engine}_{state}.ogg'
        file_path = os.path.join(DATA_DIR, state, file)

        if not os.path.exists(file_path):
            print(f'File not found: {file_path}')
            continue

        print(f'Mapping {file_path}')
        read_and_map_audio(file_path)


