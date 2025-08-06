import numpy as np

# Create a 4x4 binary image
image = np.zeros((4, 4), dtype=int)

# Choose two pixels (for example, (1, 1) and (2, 2))
pixel1 = (1, 1)
pixel2 = (2, 2)

# Set the chosen pixels to 1 in the binary image
image[pixel1] = 1
image[pixel2] = 1

print("Binary Image:")
print(image)

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

# Print results
print(f"Pixel 1: {pixel1}")
print(f"4-Adjacent Neighbors of Pixel 1: {neighbors_pixel1_4}")
print(f"8-Adjacent Neighbors of Pixel 1: {neighbors_pixel1_8}")

print(f"\nPixel 2: {pixel2}")
print(f"4-Adjacent Neighbors of Pixel 2: {neighbors_pixel2_4}")
print(f"8-Adjacent Neighbors of Pixel 2: {neighbors_pixel2_8}")
