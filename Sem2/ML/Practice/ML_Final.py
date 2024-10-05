import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 50
num_new_points = 20
x_min, x_max = 0, 1
noise_mean = 0
noise_std=0.03

# Polynomial curve fitting degree varies from 1 to 6
poly_degree = 3  # Degree of the polynomial for fitting

# Generate uniform random values for x
x = np.random.uniform(x_min, x_max, num_points)

# Compute y without noise
y = np.sin(1 + x**2)

# Generate noise
noise = np.random.normal(noise_mean, noise_std, num_points)

# Compute y with noise
y_noisy = y + noise

plt.plot(x, y, 'o', color='blue', label='Noisy data')
plt.show()

# Display multiple plots in a grid
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
