import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#part A product classification 
# Load and preprocess the dataset
categories = []
for i in range(1,21):
    categories.append(str(i))
# categories = ['1', '2', '3', ...]  # Replace with your product names
data = []
labels = []

for category in categories:
    train_folder_path = f'Data\Product Classification\{category}\Train'
    for image_name in os.listdir(train_folder_path):
        image_path = os.path.join(train_folder_path, image_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 128))
        data.append(resized)
        labels.append(category)

# Extract HOG features
hog = cv2.HOGDescriptor()
features = []
for image in data:
    hog_features = hog.compute(image)
    features.append(hog_features.flatten())

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

#Train the classification model
model = SVC()
model.fit(X_train, y_train)

# Evaluate the model on validation data
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy (Part A):", accuracy)



# #part B Product Verification/Recognition
# # Load and preprocess the dataset
# categories = ['product1', 'product2', 'product3', ...]  # Replace with your product names
# data = []
# labels = []

# for category in categories[:40]:  # Use the first 40 products for training
#     train_folder_path = f'dataset/train/{category}/'
#     for image_name in os.listdir(train_folder_path):
#         image_path = os.path.join(train_folder_path, image_name)
#         image = cv2.imread(image_path)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (64, 128))
#         data.append(resized)
#         labels.append(category)

# # Extract HOG features
# hog = cv2.HOGDescriptor()
# features = []
# for image in data:
#     hog_features = hog.compute(image)
#     features.append(hog_features.flatten())

# #Train the classification model for one/few shot learning
# model = SVC()
# model.fit(features, labels)

# # Evaluate the model on validation data
# val_data = []
# val_labels = []

# for category in categories[40:]:  # Use the remaining 20 products for validation
#     val_folder_path = f'dataset/validation/{category}/'
#     for image_name in os.listdir(val_folder_path):
#         image_path = os.path.join(val_folder_path, image_name)
#         image = cv2.imread(image_path)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (64, 128))
#         val_data.append(resized)
#         val_labels.append(category)

# val_features = []
# for image in val_data:
#     hog_features = hog.compute(image)
#     val_features.append(hog_features.flatten())

# y_pred = model.predict(val_features)
# accuracy = accuracy_score(val_labels, y_pred)
# print("Validation Accuracy (Part B):", accuracy)