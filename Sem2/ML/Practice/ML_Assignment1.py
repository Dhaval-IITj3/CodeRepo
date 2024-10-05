import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 50
x_min, x_max = 0, 1
noise_mean = 0
noise_std = 0.03
noise_std = noise_std **2

# Generate uniform random values for x
x = np.linspace(x_min, x_max, num_points)

# Compute y without noise
y_true = np.sin(1 + x**2)

# Generate noise
noise = np.random.normal(noise_mean, noise_std, num_points)

# Compute y with noise
y = y_true + noise

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Noisy data')
plt.plot(x, y_true, color='red', label='True function', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of x vs. y with Noise')
plt.legend()
plt.grid(True)
plt.show()


