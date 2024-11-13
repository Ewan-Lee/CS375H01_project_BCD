import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt


# Define the directory paths
dataset_dir = 'C:/Users/Ewan Lee/OneDrive/Desktop/NJIT Stuff/F24/CS 375 HON/Project/Dataset'
healthy_dir = os.path.join(dataset_dir, 'healthy')
tumor_dir = os.path.join(dataset_dir, 'tumor')

# Lists to store image file paths
healthy_image_data = []
tumor_image_data = []
mri_image_data=[]
labels=[]

# Check if the directories exist and load the image file paths
if os.path.exists(healthy_dir):
    for filename in os.listdir(healthy_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(healthy_dir, filename)
            img = Image.open(image_path)  # Open the image
            img_array = np.array(img)  # Convert the image to a NumPy array
            mri_image_data.append(img_array)  # Store the image data
            labels.append("Healthy")

if os.path.exists(tumor_dir):
    for filename in os.listdir(tumor_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(tumor_dir, filename)
            img = Image.open(image_path)  # Open the image
            img_array = np.array(img)  # Convert the image to a NumPy array
            mri_image_data.append(img_array)  # Store the image data
            labels.append("Tumor")


# Print the number of images loaded from each folder
print(f"Number of images: {len(mri_image_data)}")

i = random.randint(0, len(mri_image_data))

print(f"Image at index: {i}")

if mri_image_data:
    plt.imshow(mri_image_data[i])  # Use matplotlib to display the image
    plt.title(labels[i])  # Set a title for the image
    plt.axis('off')  # Turn off the axis labels
    plt.show()
