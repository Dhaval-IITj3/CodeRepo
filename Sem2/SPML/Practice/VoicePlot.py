import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


# Parameters
fs = 44100
duration = 5

# Output Filename
rec_filename = "output.wav"


print("Recording...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
sd.wait()     # Wait until recording is finished

print(f"Recording done, saving to file {rec_filename}...")
write(rec_filename, fs, recording)


# Function to compute energy
signal_energy = np.sum(np.square(recording))

# Function to compute zero-crossing count
signal_zc = np.sum(np.diff(np.signbit(recording)))

# Plot the data
time = np.linspace(0, duration, num=len(recording))

plt.figure(figsize=(12, 6))

# Plot the waveform
plt.subplot(3, 1, 1)
plt.plot(time, recording)
plt.title("Audio Waveform")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Plot the energy
plt.subplot(3, 1, 2)
plt.bar(time[0:len(recording)//len(time):len(recording)//(len(time)//10)], [signal_energy] * 10, width=0.01)
plt.title("Energy")
plt.ylabel("Energy")

# Plot the zero-crossing count
plt.subplot(3, 1, 3)
plt.bar(time[0:len(recording)//len(time):len(recording)//(len(time)//10)], [signal_zc] * 10, width=0.01)
plt.title("Zero-Crossing Count")
plt.ylabel("ZCC")

plt.tight_layout()
plt.show()
