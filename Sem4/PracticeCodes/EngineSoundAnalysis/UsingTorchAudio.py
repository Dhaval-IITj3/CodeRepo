import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def plot_audio(file_path):
    # Step 1: Load OGG file
    # file_path = 'D:\IIT_J\Repo\CodeRepo\Sem4\PracticeCodes\Resources\Before\Activa6g_Before.ogg'  # Replace with your OGG file path
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze().numpy()  # Convert to numpy for plotting
    time_axis = np.arange(len(waveform)) / sample_rate  # Time axis in seconds

    # Step 2: Plot Waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, waveform, color='blue')
    plt.title('Waveform of Engine Sound')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plot_filename=os.path.basename(file_path).split('.')[0]
    plt.savefig(f'Plots/{plot_filename}_waveform_plot.png')
    plt.close()

    # Step 3: Compute Mel-Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=64,  # Number of Mel bands
        n_fft=2048,  # FFT window size
        hop_length=512  # Hop length for time resolution
    )
    mel_spectrogram = mel_transform(torch.from_numpy(waveform).float())
    mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)  # Convert to dB

    # Step 4: Plot Mel-Spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(mel_spectrogram_db, aspect='auto', origin='lower',
               extent=[0, time_axis[-1], 0, sample_rate/2])
    plt.colorbar(label='Amplitude (dB)')
    plt.title('Mel-Spectrogram of Engine Sound')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(f'Plots/{plot_filename}_spectrogram_plot.png')
    plt.close()


if __name__ == "__main__":
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


    os.makedirs('Plots', exist_ok=True)
    # Clear any existing plots
    for plots in os.listdir('Plots'):
        os.remove(os.path.join('Plots', plots))

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

            print(f'Plotting for {file_path}')

            if file.endswith('.ogg'):
                plot_audio(file_path)


