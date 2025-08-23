import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil

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
    plt.savefig(f'Plots/waveform/{plot_filename}_waveform_plot.png')
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
    plt.savefig(f'Plots/spectrogram/{plot_filename}_spectrogram_plot.png')
    plt.close()


def plot_comparision_audios(engine_name):
    # Step 1: Load OGG file
    # file_path = 'D:\IIT_J\Repo\CodeRepo\Sem4\PracticeCodes\Resources\Before\Activa6g_Before.ogg'  # Replace with your OGG file path
    waveform_before, sample_rate_before = torchaudio.load(os.path.join(DATA_DIR, 'Before', f'{engine_name}_Before.ogg'))
    waveform_after, sample_rate_after = torchaudio.load(os.path.join(DATA_DIR, 'After', f'{engine_name}_After.ogg'))

    # Convert to mono if stereo
    if waveform_before.shape[0] > 1:
        waveform_before = torch.mean(waveform_before, dim=0, keepdim=True)

    waveform_before = waveform_before.squeeze().numpy()  # Convert to numpy for plotting
    time_axis_before = np.arange(len(waveform_before)) / sample_rate_before  # Time axis in seconds

    if waveform_after.shape[0] > 1:
        waveform_after = torch.mean(waveform_after, dim=0, keepdim=True)

    waveform_after = waveform_after.squeeze().numpy()  # Convert to numpy for plotting
    time_axis_after = np.arange(len(waveform_after)) / sample_rate_after  # Time axis in seconds

    # Step 2: Plot Before and After Waveform together in same plot
    plt.subplot(2, 1, 1)
    plt.plot(time_axis_before, waveform_before, color='blue')
    plt.title(f'{engine_name} Before')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.plot(time_axis_after, waveform_after, color='red')
    plt.title(f'{engine_name} After')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Plots/waveform/{engine_name}_comparison_plot.png')
    plt.close()

    # Step 3: Plot Before and After Mel-Spectrogram together in same plot
    mel_transform_before = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate_before,
        n_mels=64,  # Number of Mel bands
        n_fft=2048,  # FFT window size
        hop_length=512  # Hop length for time resolution
    )

    mel_spectrogram_before = mel_transform_before(torch.from_numpy(waveform_before).float())
    mel_spectrogram_db_before = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram_before)  # Convert to dB

    mel_transform_after = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate_after,
        n_mels=64,  # Number of Mel bands
        n_fft=2048,  # FFT window size
        hop_length=512  # Hop length for time resolution
    )

    mel_spectrogram_after = mel_transform_after(torch.from_numpy(waveform_after).float())
    mel_spectrogram_db_after = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram_after)  # Convert to dB

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.imshow(mel_spectrogram_db_before, aspect='auto', origin='lower',
               extent=[0, time_axis_before[-1], 0, sample_rate_before/2])
    plt.colorbar(label='Amplitude (dB)')
    plt.title(f'{engine_name} Before')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.imshow(mel_spectrogram_db_after, aspect='auto', origin='lower',
               extent=[0, time_axis_after[-1], 0, sample_rate_after/2])
    plt.colorbar(label='Amplitude (dB)')
    plt.title(f'{engine_name} After')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(f'Plots/spectrogram/{engine_name}_comparison_plot.png')
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
    shutil.rmtree(os.path.join('Plots'))

    os.makedirs('Plots/waveform', exist_ok=True)
    os.makedirs('Plots/spectrogram', exist_ok=True)

    # Step 2: Load Data (assumes folders like 'engine1_before/', etc.)
    features = []
    labels = []  # 0: before, 1: after

    for engine in engines:
        before_file_name=f'{engine}_Before.ogg'
        after_file_name=f'{engine}_After.ogg'

        if os.path.exists(os.path.join(DATA_DIR, 'Before', before_file_name)) and os.path.exists(os.path.join(DATA_DIR, 'After', after_file_name)):
            plot_comparision_audios(engine)
        else:
            if os.path.exists(os.path.join(DATA_DIR, 'Before', before_file_name)):
                plot_audio(os.path.join(DATA_DIR, 'Before', before_file_name))

            if os.path.exists(os.path.join(DATA_DIR, 'After', after_file_name)):
                plot_audio(os.path.join(DATA_DIR, 'After', after_file_name))





