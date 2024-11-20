# Import necessary libraries
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import random

# Set up the path for your dataset
dataset_path = 'C:/Users/Ewan Lee/OneDrive/Desktop/NJIT Stuff/F24/CS 375 HON/Project/CS375H01_project_BCD/Dataset'
healthy_path = os.path.join(dataset_path, 'healthy')
tumor_path = os.path.join(dataset_path, 'tumor')

# Constants
img_size = 512
input_shape = (img_size, img_size, 3)

# Helper functions for loading and preprocessing the dataset
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = tf.keras.preprocessing.image.load_img(os.path.join(folder, filename), target_size=(img_size, img_size))
        if img is not None:
            images.append(tf.keras.preprocessing.image.img_to_array(img))
    return images

# Load images for healthy and tumor categories
healthy_images = load_images_from_folder(healthy_path)
tumor_images = load_images_from_folder(tumor_path)

# Labels for the dataset
healthy_labels = ['healthy'] * len(healthy_images)
tumor_labels = ['tumor'] * len(tumor_images)

# Combine and reshape
images = np.array(healthy_images + tumor_images)
labels = np.array(healthy_labels + tumor_labels)

# Normalize the images
images = images / 255.0

# Print the number of images loaded from each folder
print(f"Number of images: {len(images)}")

i = random.randint(0, len(images))

print(images[i].shape)

print(f"Image at index: {i}")
plt.imshow(images[i])  # Use matplotlib to display the image
plt.title(labels[i])  # Set a title for the image
plt.axis('off')  # Turn off the axis labels
plt.show()
