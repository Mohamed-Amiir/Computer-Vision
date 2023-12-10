import os
import cv2
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog, ttk


def resize_image(image_path, width, height):
    original_image = Image.open(image_path)
    resized_image = original_image.resize((width, height))
    return ImageTk.PhotoImage(resized_image)


def rgb_to_hex(rgb):
    """Convert RGB tuple to hexadecimal color code."""
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def extract_features(img_path):
    # HoG
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 128))
    hog = cv2.HOGDescriptor()
    features = hog.compute(img_resized)
    return features


def calculate_similarity(img_1_path, img_2_path):
    features_1 = extract_features(img_1_path)
    features_2 = extract_features(img_2_path)
    similarity = cosine_similarity(features_1.reshape(1, -1), features_2.reshape(1, -1))

    return similarity[0][0]


# Use HoG features for classification
def classify_image(img_path):
    features = extract_features(img_path)

    # Predict category using the trained model
    predicted_label = model.predict([features])[0]

    label_result["text"] = f"Predicted Category: {predicted_label}"


def browse_file():
    # Open file selection dialog
    filename = filedialog.askopenfilename(title="Select Image")

    # Update entry with path
    entry_path.delete(0, tk.END)
    entry_path.insert(0, filename)


# Initialize empty lists for training and validation data
train_data = []
train_labels = []
validation_data = []
validation_labels = []

# Loop through each category folder
for category_folder in os.listdir("Product Classification"):
    category_path = os.path.join("Product Classification", category_folder)
    category_label = int(category_folder)

    # Load train data
    for img_path in os.listdir(os.path.join(category_path, "Train")):
        img_data = extract_features(os.path.join(category_path, "Train", img_path))
        train_data.append(img_data)
        train_labels.append(category_label)

    # Load validation data
    for img_path in os.listdir(os.path.join(category_path, "Validation")):
        img_data = extract_features(os.path.join(category_path, "Validation", img_path))
        validation_data.append(img_data)
        validation_labels.append(category_label)

# Train SVM model
model = SVC(kernel="linear")
model.fit(train_data, train_labels)

# Perform predictions on validation data
validation_predictions = model.predict(validation_data)

# Calculate accuracy
accuracy = accuracy_score(validation_labels, validation_predictions)
print("Validation Accuracy:", accuracy)

# Create main window
root = tk.Tk()