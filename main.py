import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog, ttk



# def extract_features(img_path):
#     # Load image
#     img = cv2.imread(img_path)

#     # Convert to grayscale and resize
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_resized = cv2.resize(img_gray, (128, 128))

#     # Extract features using FeatureHasher
#     hog = cv2.HOGDescriptor()
#     hog_features = hog.compute(img_resized)
#     features.append(hog_features.flatten())

#     return features.toarray()[0]
def extract_features(img_path):
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 128))

    # Create HoG descriptor
    hog = cv2.HOGDescriptor()

    # Extract features
    features = hog.compute(img_resized)

    return features

def update_result(predicted_label):
    label_result["text"] = f"Predicted Category: {predicted_label}"
def calculate_similarity(img_1_path, img_2_path):
    features_1 = extract_features(img_1_path)
    features_2 = extract_features(img_2_path)

    # Calculate cosine similarity
    similarity = cosine_similarity(features_1.reshape(1, -1), features_2.reshape(1, -1))

    return similarity[0][0]


# Use HoG features for classification
def classify_image(img_path):
    features = extract_features(img_path)

    # Predict category using the trained model
    predicted_label = model.predict([features])[0]

    update_result(predicted_label)


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

# Create main window
root = tk.Tk()
root.title("Product Classification")

# Create input field for image path
label_path = tk.Label(root, text="Image Path:")
label_path.pack()

constant_value = tk.StringVar(value="")
entry_path = tk.Entry(root,textvariable=constant_value, width=50)
entry_path.pack()

# Create button to browse for image
button_browse = tk.Button(root, text="Browse...", command=browse_file)
button_browse.pack()

# Create button to classify image
button_classify = tk.Button(root, text="Classify", command=lambda: classify_image(entry_path.get()))
button_classify.pack()


# Create label to display prediction result
label_result = tk.Label(root, text="")
label_result.pack()

# Run the main loop
root.mainloop()

