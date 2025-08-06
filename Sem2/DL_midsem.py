import numpy as np
from scipy.signal import convolve2d, correlate2d

# Define the matrices I and W
I = np.array([
    [10, 12, 11, 13, 14],
    [11, 12, 11, 87, 23],
    [23, 32, 79, 86, 8],
    [65, 78, 8, 68, 26],
    [67, 97, 92, 21, 15]
])

W = np.array([
    [-4, -8, -4],
    [0, 2, 0],
    [4, 8, 4]
])

# Perform convolution (with kernel flipping)
convolution_result = convolve2d(I, W, mode='valid')

# Perform correlation (without kernel flipping)
correlation_result = correlate2d(I, W, mode='valid')

print(convolution_result)
print(correlation_result)
