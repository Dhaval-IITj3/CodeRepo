import numpy as np
import matplotlib.pyplot as plt

# Create a 4x4 binary image
image = np.zeros((4, 4), dtype=int)

# Choose two pixels to mark (for example, (1, 1) and (2, 2))
pixel1 = (1, 1)
pixel2 = (2, 2)

# Create a color image with all pixels marked as blue
colored_image = np.zeros((4, 4, 3))  # 4x4 image with 3 color channels (RGB)
colored_image[:] = [0, 0, 1]  # Blue for all pixels

# Mark the selected pixels
colored_image[pixel1] = [1, 1, 0]  # Yellow for pixel1
colored_image[pixel2] = [1, 0, 0]  # Red for pixel2

# Display the main image
plt.imshow(colored_image)
plt.title("4x4 Binary Image with Marked Pixels")
plt.axis('off')
plt.show()


# Function to find 4-adjacent neighbors
def get_4_adjacent(pixel):
    x, y = pixel
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 4 and 0 <= ny < 4:
            neighbors.append((nx, ny))
    return neighbors


# Function to find 8-adjacent neighbors
def get_8_adjacent(pixel):
    x, y = pixel
    neighbors = []
    directions = [
        (0, 1), (1, 0), (1, 1), (0, -1),
        (-1, 0), (-1, -1), (-1, 1), (1, -1)  # right, down, down-right, left, up, up-left, up-right, down-left
    ]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 4 and 0 <= ny < 4:
            neighbors.append((nx, ny))
    return neighbors


# Get neighbors for the chosen pixels
neighbors_pixel1_4 = get_4_adjacent(pixel1)
neighbors_pixel1_8 = get_8_adjacent(pixel1)
neighbors_pixel2_4 = get_4_adjacent(pixel2)
neighbors_pixel2_8 = get_8_adjacent(pixel2)


# Function to create highlighted images
def highlight_4_neighbors(pixel, neighbors_4, color_pixel, color_4):
    highlighted_image = np.zeros((4, 4, 3))  # New image for highlighting
    highlighted_image[:] = [0, 0, 1]  # Blue for all pixels
    highlighted_image[pixel] = color_pixel  # Mark the main pixel

    # Mark 4-adjacent neighbors
    for neighbor in neighbors_4:
        highlighted_image[neighbor] = color_4

    return highlighted_image


# Function to create highlighted images
def highlight_8_neighbors(pixel, neighbors_8, color_pixel, color_8):
    highlighted_image = np.zeros((4, 4, 3))  # New image for highlighting
    highlighted_image[:] = [0, 0, 1]  # Blue for all pixels
    highlighted_image[pixel] = color_pixel  # Mark the main pixel

    # Mark 8-adjacent neighbors
    for neighbor in neighbors_8:
        highlighted_image[neighbor] = color_8

    return highlighted_image


# Create highlighted images for both pixels
highlighted_pixel1_4adj = highlight_4_neighbors(pixel1, neighbors_pixel1_4,[1, 1, 0], [0, 0.5, 0])
highlighted_pixel2_4adj = highlight_4_neighbors(pixel2, neighbors_pixel2_4,[1, 0, 0], [0.5, 0, 0.5])

highlighted_pixel1_8adj = highlight_4_neighbors(pixel1, neighbors_pixel1_8,[1, 1, 0], [0.5, 1, 0.5])
highlighted_pixel2_8adj = highlight_4_neighbors(pixel2, neighbors_pixel2_8,[1, 0, 0], [1, 0.75, 0.8])

# Display the highlighted images
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(highlighted_pixel1_4adj)
plt.title("Pixel 1 and its 4 Neighbors")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(highlighted_pixel2_4adj)
plt.title("Pixel 2 and its 4 Neighbors")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(highlighted_pixel1_8adj)
plt.title("Pixel 1 and its 8 Neighbors")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(highlighted_pixel2_8adj)
plt.title("Pixel 2 and its 8 Neighbors")
plt.axis('off')

plt.show()
