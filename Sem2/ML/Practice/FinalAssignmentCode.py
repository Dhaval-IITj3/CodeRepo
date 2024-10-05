import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Function to generate data: x, y_noisy, y_true
def generate_data(num_points, noise_std):
    x = np.linspace(0, 1, num_points)
    y_true = np.sin(1 + x ** 2)
    noise = np.random.normal(0, noise_std, num_points)
    y_noisy = y_true + noise
    return x, y_noisy, y_true

# Function to fit polynomial regression and compute MSE
def fit_and_evaluate(degree, x_train, y_train, x_test, y_test):
    poly = PolynomialFeatures(degree)
    x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
    x_test_poly = poly.transform(x_test.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    return mse_train, mse_test, model, poly


# Initial dataset
num_points = [50, 1000]
noise_std = 0.03

# Fit polynomial models and calculate MSE
degrees = range(1, 9)  # Polynomial degrees from 1 to 8
mse_train_list = []
mse_test_list = []

for num in num_points:
    x, y, y_true = generate_data(num, noise_std)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    plt.figure(figsize=(18, 12))
    plt.suptitle(f'Polynomial Regression Fit with {num} data points')

    for degree in degrees:
        mse_train, mse_test, model, poly = fit_and_evaluate(degree, x_train, y_train, x_test, y_test)
        mse_train_list.append(mse_train)
        mse_test_list.append(mse_test)

        # Predict using the model
        x_poly = poly.transform(x.reshape(-1, 1))
        y_pred = model.predict(x_poly)

        # Plot the results
        plt.subplot(3, 3, degree)
        plt.plot(x, y_true, label='True function', color='green')
        plt.scatter(x, y, label='Noisy data', color='blue', marker='.')
        plt.plot(x, y_pred, label=f'Degree {degree}', color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)



    # Plot the MSE comparison
    plt.subplot(3, 3, 9)
    plt.plot(degrees, mse_train_list, marker='o', label='Training MSE')
    plt.plot(degrees, mse_test_list, marker='o', label='Testing MSE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
