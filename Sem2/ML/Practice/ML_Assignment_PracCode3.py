import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 50
num_new_points = 20
x_min, x_max = 0, 1
noise_mean = 0
noise_std = 0.03
poly_degrees = [2, 4]  # Degrees of the polynomial for fitting

# Generate uniform random values for x
x = np.linspace(x_min, x_max, num_points)

# Compute y without noise
y_true = np.sin(1 + x ** 2)

# Generate noise
noise = np.random.normal(noise_mean, noise_std, num_points)

# Compute y with noise
y = y_true + noise

# Generate new x values for predictions
x_new = np.linspace(x_min, x_max, num_new_points)

plt.figure(figsize=(14, 10))

# Plot original noisy data
plt.scatter(x, y, color='blue', label='Noisy data')

# Plot the true function
plt.plot(x, y_true, color='red', label='True function', linestyle='--')

for degree in poly_degrees:
    # Fit a polynomial of the current degree
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)

    # Generate the polynomial fit values
    x_fit = np.linspace(x_min, x_max, 100)
    y_fit = polynomial(x_fit)

    # Predict new y values using the fitted polynomial
    y_new = polynomial(x_new)

    # Plot the polynomial fit
    plt.plot(x_fit, y_fit, label=f'Polynomial fit (degree {degree})')

    # Plot the new predictions
    plt.scatter(x_new, y_new, marker='x', label=f'Predicted new data (degree {degree})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Curve Fitting with Degrees 2 and 4')
plt.legend()
plt.grid(True)
plt.show()

# Print new x values and their corresponding predicted y values for each polynomial degree
for degree in poly_degrees:
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    y_new = polynomial(x_new)
    print(f"Polynomial degree {degree}:")
    print("New x values:")
    print(x_new)
    print("Predicted y values:")
    print(y_new)
    print()
