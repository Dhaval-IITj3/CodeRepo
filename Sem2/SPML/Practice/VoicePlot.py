import sounddevice as sd
import numpy as np
import librosa.feature
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# https://youtu.be/EycaSbIRx-0?si=f3a5NhDtTNaiWkEy

# Parameters
""" 44,100 Hz (44.1 kHz) is the sampling rate of audio CDs. """
fs = 44100          # Sample rate
duration = 5
frame_size = 1024  # Size of each frame
hop_size = 512     # Hop size for overlapping windows

# Output Filename
rec_filename = "output.wav"

def my_rmse(signal, fr_sz, hop_length):
    rmse_array = []

    # calculate rmse for each frame
    for i in range(0, len(signal), hop_length):
        rmse_current_frame = np.sqrt(sum(signal[i:i + fr_sz] ** 2) / fr_sz)
        rmse_array.append(rmse_current_frame)
    return np.array(rmse_array)

print("Recording...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
sd.wait()     # Wait until recording is finished

print(f"Recording done, saving to file {rec_filename}...")
write(rec_filename, fs, recording)

ipd.Audio(rec_filename)

# Load the recording using Librosa
loaded_recording, sr = librosa.load(rec_filename)

# Extract RMSE from the recording using Librosa
rms_energy = librosa.feature.rms(y=loaded_recording, frame_length=frame_size, hop_length=hop_size)[0]
t = librosa.frames_to_time(range(len(rms_energy)), hop_length=hop_size)

# Plot the RMSE
plt.figure(figsize=(15, 17))
ax = plt.subplot(3, 1, 1)
librosa.display.waveshow(loaded_recording, alpha=0.5)
plt.plot(t, rms_energy, color="red")
plt.title("Root Mean Square Energy using Librosa")
plt.ylim(-0.25, 0.25)
plt.xlabel("Time (s)")
plt.ylabel("RMSE")

my_rms_energy = my_rmse(loaded_recording, frame_size, hop_size)
t = librosa.frames_to_time(range(len(my_rms_energy)), hop_length=hop_size)
plt.subplot(3, 1, 2)
librosa.display.waveshow(loaded_recording, alpha=0.5)
plt.plot(t, my_rms_energy, color="red")
plt.title("Root Mean Square Energy using my custom method")
plt.ylim(-0.25, 0.25)
plt.xlabel("Time (s)")
plt.ylabel("RMSE")

# Zero corssing rate with Librosa
print("Total zero crossing count: ", ((loaded_recording[:-1] * loaded_recording[1:]) < 0).sum())

zcr = librosa.feature.zero_crossing_rate(y=loaded_recording, frame_length=frame_size, hop_length=hop_size)[0]
t = librosa.frames_to_time(range(len(zcr)), hop_length=hop_size)

plt.subplot(3, 1, 3)
librosa.display.waveshow(loaded_recording, alpha=0.5)
plt.plot(t, zcr, color="red")
plt.title("Zero Crossing Rate using Librosa")
plt.ylim(-1, 1)
plt.xlabel("Time (s)")
plt.ylabel("ZCR")
plt.show()


