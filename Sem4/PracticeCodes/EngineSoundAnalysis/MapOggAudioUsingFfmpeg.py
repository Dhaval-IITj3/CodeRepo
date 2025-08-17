import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ffmpeg
import soundfile as sf
import io


def read_and_map_audio(input_file):
    print(f"Reading file: {input_file}")

    # Convert .ogg to raw PCM audio in memory using ffmpeg
    try:
        out, _ = (
            ffmpeg
            .input(input_file)
            .output('pipe:', format='wav')
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())
        return None

    # Read audio data from memory buffer
    audio_data, sample_rate = sf.read(io.BytesIO(out))

    if audio_data.ndim > 1:
        channels = audio_data.shape[1]
    else:
        channels = 1

    total_samples = audio_data.shape[0]

    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Total Samples: {total_samples}")

    # If stereo, use first channel for plotting
    if channels > 1:
        audio_data = audio_data[:, 0]

    # Plot waveform
    plt.figure(figsize=(10, 4))
    plt.plot(audio_data[:10000])  # Plot first 10,000 samples
    plt.title(f"Waveform of {Path(input_file).name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    return audio_data


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


