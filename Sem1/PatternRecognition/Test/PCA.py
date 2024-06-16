"""
Given the 2-dimensional dataset with points {(1, 2),(2, 3),(3, 4),(4, 5)},
perform Principal Component Analysis (PCA) to find the principal components.
Specifically, compute the covariance matrix,
find the eigenvalues and eigenvectors, and determine the direction of the principal components.
Assume you have already centered the data by subtracting the mean of each feature.
After performing PCA, if you project the original dataset onto the first principal component,
what will be the projected values for each point?
You can use python function for calculation.
"""


import numpy as np

# Given input: Centered data points
X_centered = np.array([[1, 2],
                        [2, 3],
                        [3, 4],
                        [4, 5]])

print("Centered Data:\n", X_centered)

# Compute the covariance matrix
cov_matrix = np.cov(X_centered.T)
print("\nCovariance Matrix:\n", cov_matrix)

# Compute eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)


# Step 3: Sort the eigenvectors by decreasing eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]

# Sort the eigenvalues in descending order
eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# The direction of the first principal component is the eigenvector with the highest eigenvalue
first_principal_component = sorted_eigenvectors[:, 0]

# Step 4: Project the original data onto the first principal component
projected_data = X_centered.dot(first_principal_component)

print("\nFirst Principal Component:\n", first_principal_component)
print("\nProjected Data:\n", projected_data)
