import numpy as np
import matplotlib.pyplot as plt

# --- 1. Create a sample time-domain signal ---
SAMPLE_RATE = 1000 # Hz
DURATION = 1       # Seconds
N = SAMPLE_RATE * DURATION # Total number of samples

# Time array
t = np.linspace(0., DURATION, N, endpoint=False) #

# Create a signal composed of two sine waves at 5 Hz and 20 Hz
freq1 = 5
freq2 = 20
amplitude1 = 1.0
amplitude2 = 0.5
signal = amplitude1 * np.sin(2 * np.pi * freq1 * t) + amplitude2 * np.cos(2 * np.pi * freq2 * t)

# --- 2. Perform the Forward FFT (DFT) ---
# X contains the complex frequency components
X = np.fft.fft(signal)
# Generate the corresponding frequency bins for plotting
freqs = np.fft.fftfreq(N, 1/SAMPLE_RATE)

# --- 3. Perform the Inverse IFFT (IDFT) ---
# x_rec is the reconstructed time-domain signal (complex numbers with negligible imaginary part)
x_rec = np.fft.ifft(X)

# Since the input signal was real, the reconstructed signal should also be real.
# We take the real part to discard any small imaginary components due to floating-point errors.
x_rec_real = np.real(x_rec)

# --- 4. Plot the results ---

plt.figure(figsize=(12, 6))

# Plot the original signal
plt.subplot(121)
plt.plot(t, signal, 'b', label='Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Time Domain Signal')
plt.legend()
plt.grid(True)

# Plot the reconstructed signal using IFFT
plt.subplot(122)
plt.plot(t, x_rec_real, 'r--', label='Reconstructed Signal (IDFT)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Signal Reconstructed via IDFT')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
