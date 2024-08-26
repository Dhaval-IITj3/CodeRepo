import numpy as np
import matplotlib.pyplot as plot

# Define total number of data points
num_data_points = 50

# Generate random uniform x values between 0 and 1
x = np.random.uniform(0,1,num_data_points)

# Calculate y values using given formula y = sin(1+x^2)
y = np.sin(1 + (x**2))

# Add noise with normal distribution N(0,0.03^2)
noise = np.random.normal(0, 0.03, num_data_points)
noisy_y = y + noise

# Plot the data
plot.scatter(x, noisy_y, label='Noisy Data')
plot.plot(x, y, color='red', label='Original Data without Noise')
plot.xlabel('x')
plot.ylabel('y')
plot.legend()
plot.show()
