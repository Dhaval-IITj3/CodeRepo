import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 50
num_new_points = 20
x_min, x_max = 0, 1
noise_mean = 0
noise_std = 0.03
poly_degrees = [1, 2, 3, 4, 5, 6, 7, 8]  # Degrees of the polynomial for fitting

# Colors
red = '#FF0000'
blue = '#5555EB'
pink = '#FFBBBB'
dark_grey = '#333333'


# Generate uniform random values for x
x = np.linspace(x_min, x_max, num_points)

# Compute y without noise
y_true = np.sin(1 + x ** 2)

# Generate noise
noise = np.random.normal(noise_mean, noise_std, num_points)

# Compute y with noise
y = y_true + noise

# Generate new x values for predictions
x_new = np.random.uniform(x_min, x_max, num_new_points)

# Plotting in a grid layout
plt.figure(figsize=(18, 12))

for i, degree in enumerate(poly_degrees):
    # Fit a polynomial of the current degree
    coefficients = np.polyfit(x, y, degree)
    print(f'Coefficients: {coefficients}')

    polynomial = np.poly1d(coefficients)
    print(f'Polynomial: {polynomial}')

    # Generate the polynomial fit values
    x_fit = np.linspace(x_min, x_max, 1000)
    y_fit = polynomial(x_fit)

    # Predict new y values using the fitted polynomial
    y_new = polynomial(x_new)

    # Plot the polynomial fit in a subplot
    plt.subplot(2, 4, i+1)  # 2 rows, 3 columns
    plt.scatter(x, y, color=blue, label='Noisy data', marker='.')
    plt.plot(x_fit, y_fit, label=f'Polynomial fit (degree {degree})', color=pink, linestyle='-', linewidth=1.5)
    plt.scatter(x_new, y_new, marker='x', color=red, label='Predicted new data')
    plt.plot(x, y_true, color=dark_grey, label='True function', linestyle='dotted', linewidth=1.5)
    plt.title(f'Polynomial Degree {degree}')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.legend()
    plt.grid(True)

plt.tight_layout()
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
