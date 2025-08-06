import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_dir = "Resources"
image_filenames = []
processed_images = {}
k_values = [2, 5, 10]

# Iterate over files in directory
for name in os.listdir(image_dir):
    if os.path.isfile(os.path.join(image_dir, name)):
        image_filenames.append(os.path.join(image_dir, name))

if os.path.isfile('Resources\\eight.tif'):
    image_filenames.remove("Resources\\eight.tif")

# Function to apply contrast stretching with a given K value
def contrast_stretching(image, K):
    return np.clip(K * image, 0, 255).astype(np.uint8)


# Function for logarithmic transformation
def logarithmic_transform(image, c=1):
    c = 255 / np.log(1 + np.max(image))
    return np.array(c * (np.log(1 + image)), dtype=np.uint8)


# Function to perform histogram equalization
def histogram_equalization(image):
    return cv2.equalizeHist(image)


for img_name in image_filenames:
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    processed_images[img_name] = {
        "original": img,
        "k2_contrast": contrast_stretching(img, 2),
        "k5_contrast": contrast_stretching(img, 5),
        "k10_contrast": contrast_stretching(img, 10),
        "log": logarithmic_transform(img),
        "equalized": histogram_equalization(img)
    }

for name, variants in processed_images.items():
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Result for {name}')

    axs[0, 0].imshow(variants["original"], cmap="gray")
    axs[0, 0].set_title("Original")
    axs[0, 1].imshow(variants["k2_contrast"], cmap="gray")
    axs[0, 1].set_title("K = 2")
    axs[0, 2].imshow(variants["k5_contrast"], cmap="gray")
    axs[0, 2].set_title("K = 5")
    axs[1, 0].imshow(variants["k10_contrast"], cmap="gray")
    axs[1, 0].set_title("K = 10")
    axs[1, 1].imshow(variants["log"], cmap="gray")
    axs[1, 1].set_title("Logarithmic Transformation")
    axs[1, 2].imshow(variants["equalized"], cmap="gray")
    axs[1, 2].set_title("Histogram Equalization")
    plt.show()