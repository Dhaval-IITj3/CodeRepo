# Given the dataset of images and labels, find the accuracy of the model
# Divide the dataset into training and testing
# Use the training dataset to train the model
# Use the testing dataset to test the model

import os
import random
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

### ** Set the Directory Paths **
dataset_dirname="LogoDataset"
test_dataset_dirname="TestDataset"
train_dataset_dirname="TrainDataset"

base_dirpath = os.path.dirname(__file__)
dataset_dirpath = os.path.join(base_dirpath, dataset_dirname)
test_dataset_dirpath = os.path.join(base_dirpath, test_dataset_dirname)
train_dataset_dirpath = os.path.join(base_dirpath, train_dataset_dirname)

test_images_filenames = []
train_images_filenames = []

# If the directories don't exist, create them
if not os.path.exists(test_dataset_dirpath):
    os.mkdir(test_dataset_dirpath)

if not os.path.exists(train_dataset_dirpath):
    os.mkdir(train_dataset_dirpath)

### ** Split the given images into the training and testing dataset **
# Get list of all sub directories
all_dirs = os.listdir(dataset_dirpath)
for dir in all_dirs:
    src_dir = os.path.join(dataset_dirpath, dir)
    test_dest_dirpath = os.path.join(test_dataset_dirpath, dir)
    train_dest_dirpath = os.path.join(train_dataset_dirpath, dir)

    all_files = os.listdir(src_dir)
    random.shuffle(all_files)

    if not os.path.exists(test_dest_dirpath):
        os.mkdir(test_dest_dirpath)

        # Copy files from src_dir to test_dest_dir
        for file in all_files[:2]:
            src_file = os.path.join(src_dir, file)
            test_dest_file = os.path.join(test_dest_dirpath, file)
            shutil.copy(src_file, test_dest_file)
            test_images_filenames.append(test_dest_file)

    if not os.path.exists(train_dest_dirpath):
        os.mkdir(train_dest_dirpath)

        # Copy remaining files from src_dir to train_dest_dir
        for file in all_files[2:]:
            src_file = os.path.join(src_dir, file)
            train_dest_file = os.path.join(train_dest_dirpath, file)
            shutil.copy(src_file, train_dest_file)
            train_images_filenames.append(train_dest_file)

Logos = {}
Classes = []
for dir in os.listdir(train_dataset_dirpath):
    Classes.append(dir)
    for img_file in os.listdir(os.path.join(train_dataset_dirpath, dir)):
        img = os.path.join(train_dataset_dirpath, dir, img_file)
        gray = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # Resize the image to 250x250 pixels
        # Resizing is important for the model as np arrays are of fixed size
        Logos[img_file] = cv2.resize(gray, (250, 250))

# Display all classes
print("Length of Classes: ", len(Classes))
print("Classes: ", Classes)

# Display all images
def display_all_images():
    print ("Total number of Logo Images: ", len(Logos))
    total_cols = 5
    total_rows = len(Logos) // total_cols + 1
    fig, axes = plt.subplots(total_rows, total_cols, figsize=(10, 10))
    logo_images = list(Logos.values())

    for i in range(total_rows):
        for j in range(total_cols):

            # Skip if there are no more images
            if i * 5 + j >= len(logo_images):
                axes[i, j].axis('off')
                continue

            axes[i, j].imshow(logo_images[i * 5 + j], cmap='gray')
            axes[i, j].axis('off')

    plt.show()


logomatrix = []
logolabels = []
for key, value in Logos.items():
    logomatrix.append(value.flatten())
    logolabels.append(key.split('.')[0])

logomatrix = np.array(logomatrix)
y = np.array(logolabels)

from sklearn.decomposition import PCA

pca = PCA().fit(logomatrix)

# Take the first K principal components as eigenfaces
n_components = len(Logos)
eigenfaces = pca.components_[:n_components]
logoshape = list(Logos.values())[0].shape

# Show the first 16 eigenfaces
# fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
# for i in range(16):
#     axes[i%4][i//4].imshow(eigenfaces[i].reshape(logoshape), cmap="gray")
# plt.show()

# Generate weights as a KxN matrix where K is the number of eigenfaces and N the number of samples
weights = eigenfaces @ (logomatrix - pca.mean_).T
# weights = []
# for i in range(logomatrix.shape[0]):
#     weight = []
#     for j in range(n_components):
#         w = eigenfaces[j] @ (logomatrix[i] - pca.mean_)
#         weight.append(w)
#     weights.append(weight)

for dir in os.listdir(test_dataset_dirname):
    for query in os.listdir(os.path.join(test_dataset_dirname, dir)):
        query_img_file = os.path.join(test_dataset_dirname, dir, query)
        gray = cv2.imdecode(np.fromfile(query_img_file, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        query_img = cv2.resize(gray, (250, 250))


        # Perform PCA
        weights = eigenfaces @ (query_img - pca.mean_).T
        print("Shape of weights: ", weights.shape)




