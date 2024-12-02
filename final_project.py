import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set up the path for your dataset
dataset_path = 'C:/Users/Ewan Lee/OneDrive/Desktop/NJIT Stuff/F24/CS 375 HON/Project/CS375H01_project_BCD/Dataset'
healthy_path = os.path.join(dataset_path, 'healthy')
tumor_path = os.path.join(dataset_path, 'tumor')

# Constants
img_size = 128
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
healthy_labels = [0] * len(healthy_images)
tumor_labels = [1] * len(tumor_images)

# Combine and reshape
images = np.array(healthy_images + tumor_images)
labels = np.array(healthy_labels + tumor_labels)

# Normalize the images
images = images / 255.0

# Split the training and testing data, images being the x and labels being the y
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape the data
X_train = np.mean(X_train, axis=-1, keepdims=True)
X_test = np.mean(X_test, axis=-1, keepdims=True)

# Defining the model and adding Convolution, Pooling, Connected, and Output layer
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(img_size,img_size,1)))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(2,activation='softmax'))

# Compiling the model with loss function, optimizer, and metric for function
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Fitting the model and evaluating it to the data
trained_model = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Display the loss and the accuracy of the model
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# Plot the training and validation loss in the first subplot
axes[0].plot(trained_model.history['loss'], label='Train Loss')
axes[0].plot(trained_model.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss')
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend(loc='upper right')

# Plot the training and validation accuracy in the second subplot
axes[1].plot(trained_model.history['accuracy'], label='Train Accuracy')
axes[1].plot(trained_model.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Model Accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend(loc='upper right')

# Adjust layout for better appearance
plt.tight_layout()

# Show the combined plot
plt.show()
# Making predictions and storing the results, comparing percentages            
pred=np.round(model.predict(X_test))
result=[]
for i in range(len(pred)):
    if (pred[i][0]==0):
        result.append("Tumor")
    else:
        result.append("Healthy")
correct=0
for i in range(len(result)):
    if (result[i]=="Healthy" and y_test[i]==0) or (result[i]=="Tumor" and y_test[i]==1):
        correct+=1
percent=correct/len(result) * 100  
print(f"Percent of correct predictions: {percent}%")

# Prints 10 random images to demonstrate prediction vs actual label
i = random.randint(0, len(X_test))
# Show sample images of both healthy and tumor
plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(2, 10 // 2, i + 1)
    plt.imshow(X_test[i])  # Use matplotlib to display the image
    label = "Healthy" if y_test[i]==0 else "Tumor"
    plt.title(f"Image at index: {i}\nPredicted: {result[i]} \nActual: {label} ")  # Set a title for the image
    plt.axis('off')
    i = random.randint(0, len(X_test))
plt.show()
