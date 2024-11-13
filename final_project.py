import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt


# Define the directory paths
dataset_dir = 'C:/Users/Ewan Lee/OneDrive/Desktop/NJIT Stuff/F24/CS 375 HON/Project/Dataset'
healthy_dir = os.path.join(dataset_dir, 'healthy')
tumor_dir = os.path.join(dataset_dir, 'tumor')

mri_image_data=[]
img_labels=[]

# Check if the directories exist and load the image file paths
if os.path.exists(healthy_dir):
    for filename in os.listdir(healthy_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(healthy_dir, filename)
            img = Image.open(image_path)  # Open the image
            img_array = np.array(img)  # Convert the image to a NumPy array
            mri_image_data.append(img_array)  # Store the image data
            img_labels.append("Healthy")

if os.path.exists(tumor_dir):
    for filename in os.listdir(tumor_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(tumor_dir, filename)
            img = Image.open(image_path)  # Open the image
            img_array = np.array(img)  # Convert the image to a NumPy array
            mri_image_data.append(img_array)  # Store the image data
            img_labels.append("Tumor")

temp = list(zip(mri_image_data, img_labels))
random.shuffle(temp)
images, labels = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
images, labels = list(images), list(labels)

i = random.randint(0, len(mri_image_data))

print(f"Image at index: {i}")

while True:
    plt.imshow(images[i])  # Use matplotlib to display the image
    plt.title(labels[i])  # Set a title for the image
    plt.axis('off')  # Turn off the axis labels
    plt.show()
    i = random.randint(0, len(mri_image_data))

'''random.shuffle()
images=np.array(mri_image_data)
labels=np.array(img_labels)
s = np.arange(labels.size)
np.random.shuffle(s)
# shuffle sample
images = images[s]
labels = labels[s]'''