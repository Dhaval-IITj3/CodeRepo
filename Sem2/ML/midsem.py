# Answer 3
print("Starting Answer 3")
# Input values: x = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
# Output values: t = [-4.9, -3.5, -2.8, 0.8, 0.3, -1.6, -1.3, 0.5, 2.1, 2.9, 5.6]

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Given data
x = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
t = np.array([-4.9, -3.5, -2.8, 0.8, 0.3, -1.6, -1.3, 0.5, 2.1, 2.9, 5.6])

# Parameters for Gaussian basis functions
M = 4  # Number of basis functions
sigma = 1.0  # Variance of each Gaussian basis function

# Initial guess for parameters
initial_guess = [-0.5, 0.5, 1, 1.5] * 2  # Adjusted initial guess


# Define Gaussian basis function
def gaussian_basis(x, mu, sigma=1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Define the curve function as a linear combination of Gaussian basis functions
def curve_function(x, *params):
    M = len(params) // 2
    mus = params[:M]
    weights = params[M:]
    return sum(w * gaussian_basis(x, mu) for mu, w in zip(mus, weights))


# Fit the curve using curve_fit with Trust Region Reflective algorithm
params, _ = curve_fit(curve_function, x, t, p0=initial_guess, method='trf', max_nfev=10000000)
print("Optimized parameters:", params)

# Plot the fitted curve
# Generate points to plot the curve
x_plot = np.linspace(-1, 1, 100)
y_plot = curve_function(x_plot, *params)
plt.plot(x, t, 'ro', label='Data')
plt.plot(x_plot, y_plot, 'b-', label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Curve Fitting with Gaussian Basis Functions')
plt.legend()
plt.grid(True)
plt.show()
