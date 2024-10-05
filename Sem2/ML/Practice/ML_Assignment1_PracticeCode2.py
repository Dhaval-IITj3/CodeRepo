import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 50
num_new_points = 20
x_min, x_max = 0, 1
noise_mean = 0
noise_variance = 0.03
noise_std=0.03
poly_degree = 3  # Degree of the polynomial for fitting

# Generate uniform random values for x
x = np.linspace(x_min, x_max, num_points)

# Compute y without noise
y_true = np.sin(1 + x**2)

# Generate noise
noise = np.random.normal(noise_mean, noise_std, num_points)

# Compute y with noise
y = y_true + noise

# Fit a polynomial to the data
coefficients = np.polyfit(x, y, poly_degree)
polynomial = np.poly1d(coefficients)

# Generate new x values
x_new = np.linspace(x_min, x_max, num_new_points)

# Predict new y values using the fitted polynomial
y_new = polynomial(x_new)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot original noisy data
plt.scatter(x, y, color='blue', label='Noisy data')

# Plot the true function
plt.plot(x, y_true, color='red', label='True function', linestyle='--')

# Plot the polynomial fit
x_fit = np.linspace(x_min, x_max, 100)
y_fit = polynomial(x_fit)
plt.plot(x_fit, y_fit, color='green', label=f'Polynomial fit (degree {poly_degree})')

# Plot the new predictions
plt.scatter(x_new, y_new, color='purple', marker='x', label='Predicted new data')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Curve Fitting')
plt.legend()
plt.grid(True)
plt.show()

# Print new x and y values
print("New x values:")
print(x_new)
print("Predicted y values:")
print(y_new)
