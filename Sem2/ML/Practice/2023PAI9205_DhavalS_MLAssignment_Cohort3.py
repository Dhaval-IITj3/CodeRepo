import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Function to generate dataset
def generate_dataset(n_pts):
    np.random.seed(42)
    x = np.random.uniform(0, 1, n_pts)
    noise = np.random.normal(0, 0.03, n_pts)
    y_noisy = np.sin(1 + x ** 2) + noise
    return x, y_noisy


# Function to perform polynomial regression and compute MSE
def polynomial_regression(x_trn, y_trn, x_tst, y_tst, dgr):
    poly = PolynomialFeatures(dgr)
    x_train_poly = poly.fit_transform(x_trn.reshape(-1, 1))
    x_test_poly = poly.transform(x_tst.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_train_poly, y_trn)

    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)

    mse_trn_ret = mean_squared_error(y_trn, y_train_pred)
    mse_tst_ret = mean_squared_error(y_tst, y_test_pred)

    return y_train_pred, y_test_pred, mse_trn_ret, mse_tst_ret


# Function to visualize polynomial fits
def visualize_polynomial_fits(x_train_, y_train_, x_test_, y_test_, degrees_):
    plt.figure(figsize=(14, 10))

    mse_train_list = []
    mse_test_list = []

    for k, dgr in enumerate(degrees_):
        y_train_pred, y_test_pred, mse_trn, mse_tst = polynomial_regression(x_train_, y_train_, x_test_, y_test_, dgr)

        mse_train_list.append(mse_trn)
        mse_test_list.append(mse_tst)

        # Subplot for polynomial fit
        plt.subplot(3, 3, k + 1)
        plt.scatter(x_train, y_train, label='Training data', color='blue', marker='.')
        plt.scatter(x_test, y_test, label='Testing data', color='red', marker='x')
        plt.scatter(x_train, y_train_pred, label=f'Pred Data {dgr}', color='green', marker='.')
        plt.legend()
        plt.title(f'Degree {dgr}')
        plt.xlabel('x')
        plt.ylabel('y')

    plt.subplot(3, 3, 9)
    plt.plot(degrees, mse_train_list, marker='x', label='Train MSE')
    plt.plot(degrees, mse_test_list, marker='o', label='Test MSE')
    plt.title('MSE vs Polynomial Degree')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    return mse_train_list, mse_test_list


# Main code
n_points_list = [50, 1000]

degrees = range(1, 8)
mse_train = []
mse_test = []

# Generate and visualize for 50 data points
for i, n_points in enumerate(n_points_list):
    x, y = generate_dataset(n_points)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    tr, ts = visualize_polynomial_fits(x_train, y_train, x_test, y_test, degrees)

    mse_train.append(tr)
    mse_test.append(ts)

# Generate and visualize for mse data
plt.title('MSE vs Polynomial Degree for Different Data Points')

for i, n_points in enumerate(n_points_list):
    plt.plot(degrees, mse_train[i], marker='.', label=f'Train MSE {n_points} points', color=color[i*2], linestyle='dashed')
    plt.plot(degrees, mse_test[i], marker='x', label=f'Test MSE {n_points} points', color=color[i*2+1])

plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

