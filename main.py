import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog, ttk
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

global train_data
global train_labels
global validation_data
global validation_labels
global model
model = SVC(kernel="linear", C = 0.7)

train_data = []
train_labels = []
validation_data = []
validation_labels = []
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



# #############
# train_data = np.array(train_data)
# train_labels = np.array(train_labels)

# indices = np.arange(len(train_data))
# np.random.shuffle(indices)
# train_labels = train_data[indices]
# train_labels = train_labels[indices]
# #############


def calculate_accuracy():
    
    train_data_sh, train_labels_sh = shuffle(train_data, train_labels, random_state=42)

    model.fit(train_data_sh, train_labels_sh)


    train_predictions = model.predict(train_data_sh)
    test_predictions = model.predict(validation_data)

    train_accuracy = accuracy_score(train_labels_sh, train_predictions)
    svm_train_accuracy = np.mean(cross_val_score(model, train_data_sh, train_labels_sh, cv=5))

    test_accuracy = accuracy_score(validation_labels, test_predictions)

    label_accuracy["text"] = f"Test Accuracy: %{test_accuracy*100:.2f}"
    label_crossAccuracy["text"] = f"Cross Validation Accuracy: %{svm_train_accuracy*100:.2f}.2f"
    label_trainAccuracy["text"] = f"Train Accuracy: %{train_accuracy*100:.2f}"

    #return train_accuracy, test_accuracy


# def calculate_accuracy():

#     model = SVC(kernel="linear")
#     model.fit(train_data, train_labels)
#     validation_predictions = model.predict(validation_data)
#     accuracy = accuracy_score(validation_labels, validation_predictions)
#     label_accuracy["text"] = f"Accuracy: %{accuracy*100}"






### test > train     ok --> COULD BE BETTER

### test < train     Overfitting




#Initialize empty lists for training and validation data







###################  LOADING DATA  #####################
for category_folder in os.listdir("Product Classification"):
    category_path = os.path.join("Product Classification", category_folder)
    category_label = int(category_folder)

    # Load train data
    for img_path in os.listdir(os.path.join(category_path, "augmented_train")):
        img_data = extract_features(os.path.join(category_path, "augmented_train", img_path))
        train_data.append(img_data)
        train_labels.append(category_label)

    # Load validation data
    for img_path in os.listdir(os.path.join(category_path, "Validation")):
        img_data = extract_features(os.path.join(category_path, "Validation", img_path))
        validation_data.append(img_data)
        validation_labels.append(category_label)














# Train SVM model
# model = SVC(kernel="linear")
# model.fit(train_data, train_labels)
# # Perform predictions on validation data
# validation_predictions = model.predict(validation_data)

# # Calculate accuracy
# accuracy = accuracy_score(validation_labels, validation_predictions)
# # accuracy = accuracy_score(y_val, y_pred)
# print("Validation Accuracy (Part A):", accuracy)
#Create main window
root = tk.Tk()
root.geometry("750x650")
root.resizable(width=False, height=False)

root.title("Product Classification")
# Provide the absolute path to your image file
image_path = r"C:\Users\lenovo\Desktop\My-Github\Computer-Vision\img2.jpeg"
photo = resize_image(image_path, width=750, height=650)


label = tk.Label(root, image=photo)
label.place(x=0, y=0, relwidth=1, relheight=1)  # Fit the label to the window size




# Create input field for image path
# label_path = tk.Label(root, text="Image Path:")
# label_path.place(x=300,y=10)
# label_path.grid(row=0, column=1, padx=10, pady=10)
#label_path.pack()

constant_value = tk.StringVar(value="")
entry_path = tk.Entry(root,textvariable=constant_value, width=50)
entry_path.place(x=800,y=20)
#entry_path.pack()
rgb_tuple = (171,134,92)  
# Create button to browse for image
#button_browse = tk.Button(root, text="Browse...",height=2,width=10, bg=rgb_to_hex(rgb_tuple), fg="white")
button_browse = tk.Button(root, text="Browse...",height=2,width=10, bg=rgb_to_hex(rgb_tuple), fg="white",command=browse_file)
button_browse.place(x=610,y=110)

#button_browse.pack()

# Create button to classify image
#button_classify = tk.Button(root, text="Classify",height=3,width=12, bg= rgb_to_hex(rgb_tuple), fg="white")
button_classify = tk.Button(root, text="Classify",height=3,width=12,bg= rgb_to_hex(rgb_tuple), fg="white", command=lambda: classify_image(entry_path.get()))
button_classify.place(x=579,y=176)
#button_classify.pack()

button_accuracy = tk.Button(root, text="Accuracy",height=2,width=10,bg= rgb_to_hex(rgb_tuple), fg="white", command=calculate_accuracy)
button_accuracy.place(x=457,y=140)
#button_classify.pack()


# Create label to display prediction result
label_result = tk.Label(root, text="",height=2,width=26,bg= "white", fg="black")
label_result.place(x=495,y=503)
#label_result.pack()

label_accuracy = tk.Label(root, text="",bg= "green", fg="white")
label_accuracy.place(x=490,y=468)


label_trainAccuracy = tk.Label(root, text="",bg= "green", fg="white")
label_trainAccuracy.place(x=490,y=440)


label_crossAccuracy = tk.Label(root, text="",bg= "green", fg="white")
label_crossAccuracy.place(x=490,y=410)
# #label_accuracy.pack()

# Run the main loop
root.mainloop()

