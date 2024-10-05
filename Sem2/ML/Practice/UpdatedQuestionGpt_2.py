import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Function to generate the dataset
def generate_data(num_points, noise_std):
    x = np.linspace(0, 1, num_points)
    y_true = np.sin(1 + x ** 2)
    noise = np.random.normal(0, noise_std, num_points)
    y = y_true + noise
    return x, y


# Function to fit polynomial regression and compute MSE
def fit_polynomial_regression(x_train, y_train, x_test, y_test, degree):
    poly_features = PolynomialFeatures(degree)
    x_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
    x_test_poly = poly_features.transform(x_test.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    return model, poly_features, mse_train, mse_test


# Function to plot polynomial fits and MSE
def plot_fits_and_mse(degrees, x_train, y_train, x_test, y_test, x_new, y_new_pred_list, mse_train_list, mse_test_list):
    plt.figure(figsize=(18, 24))
    for i, degree in enumerate(degrees):
        plt.subplot(len(degrees), 1, i + 1)

        plt.scatter(x_train, y_train, color='blue', label='Training data')
        plt.scatter(x_test, y_test, color='orange', label='Testing data')
        plt.plot(x_new, y_new_pred_list[i], color='red', label=f'Polynomial degree {degree}')

        plt.title(f'Degree {degree} - MSE Train: {mse_train_list[i]:.4f}, MSE Test: {mse_test_list[i]:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# Function to plot MSE comparison
def plot_mse_comparison(degrees, mse_train_list, mse_test_list, title):
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, mse_train_list, marker='o', label='Training MSE')
    plt.plot(degrees, mse_test_list, marker='o', label='Testing MSE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function to run the analysis
def analyze_polynomial_regression(num_points_list, degrees):
    mse_train_50, mse_test_50 = [], []
    mse_train_1000, mse_test_1000 = [], []

    for num_points in num_points_list:
        # Generate data
        x, y = generate_data(num_points, noise_std=0.03)

        # Split the dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        mse_train_list, mse_test_list, y_new_pred_list = [], [], []

        # Fit models and calculate MSE for each degree
        for degree in degrees:
            model, poly_features, mse_train, mse_test = fit_polynomial_regression(x_train, y_train, x_test, y_test,
                                                                                  degree)
            mse_train_list.append(mse_train)
            mse_test_list.append(mse_test)

            # Predict on new x values
            x_new = np.linspace(0, 1, 100)
            x_new_poly = poly_features.transform(x_new.reshape(-1, 1))
            y_new_pred = model.predict(x_new_poly)
            y_new_pred_list.append(y_new_pred)

        # Plot the polynomial fits and MSEs
        plot_fits_and_mse(degrees, x_train, y_train, x_test, y_test, x_new, y_new_pred_list, mse_train_list,
                          mse_test_list)

        # Store MSEs for comparison
        if num_points == 50:
            mse_train_50 = mse_train_list
            mse_test_50 = mse_test_list
        else:
            mse_train_1000 = mse_train_list
            mse_test_1000 = mse_test_list

        # Plot MSE comparison for current dataset size
        plot_mse_comparison(degrees, mse_train_list, mse_test_list, title=f'MSE Comparison (Data Points: {num_points})')

    # Compare MSEs between the two datasets (50 and 1000 points)
    plot_mse_comparison(degrees, mse_train_50, mse_test_50, title='MSE Comparison for 50 Data Points')
    plot_mse_comparison(degrees, mse_train_1000, mse_test_1000, title='MSE Comparison for 1000 Data Points')
    plt.figure(figsize=(12, 8))
    plt.plot(degrees, mse_train_50, marker='o', label='Training MSE (50 points)')
    plt.plot(degrees, mse_test_50, marker='o', label='Testing MSE (50 points)')
    plt.plot(degrees, mse_train_1000, marker='o', linestyle='--', label='Training MSE (1000 points)')
    plt.plot(degrees, mse_test_1000, marker='o', linestyle='--', label='Testing MSE (1000 points)')

    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE Comparison: 50 vs 1000 Data Points')
    plt.legend()
    plt.grid(True)
    plt.show()


# Run the analysis with datasets of 50 and 1000 points
degrees = list(range(0, 7))
analyze_polynomial_regression([50, 1000], degrees)
