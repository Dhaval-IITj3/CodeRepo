import numpy as np
import matplotlib.pyplot as plt

# Create a 4x4 binary image (0s and 1s)
image = np.zeros((4, 4), dtype=int)

# Choose two pixels to mark in red (for example, (1, 1) and (2, 2))
pixel1 = (1, 1)
pixel2 = (2, 2)

# Set the chosen pixels to 1 in the binary image
image[pixel1] = 1
image[pixel2] = 1

# Create a color image with blue background
colored_image = np.zeros((4, 4, 3))  # 4x4 image with 3 color channels (RGB)
colored_image[image == 1] = [1, 0, 0]  # Red for marked pixels

# Mark the rest of the image as blue
colored_image[image == 0] = [0, 0, 1]  # Blue for background

# Display the image
plt.imshow(colored_image)
plt.title("4x4 Binary Image with Marked Pixels")
plt.axis('off')  # Turn off the axis
plt.show()
