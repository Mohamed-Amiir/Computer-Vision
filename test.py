import os
import cv2
import numpy as np

# Input and output folders
input_folder = "C:\\Users\\lenovo\\Desktop\\My-Github\\Computer-Vision\\Product Classification\\1\\Train"
output_folder = "C:\\Users\\lenovo\\Desktop\\My-Github\\Computer-Vision\\Product Classification\\1\\Train"
# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Data augmentation function
def apply_augmentation(image):
    # Randomly crop the image
    crop_fraction = np.random.uniform(0.8, 1.0)
    height, width = image.shape[:2]
    h_crop = int(height * crop_fraction)
    w_crop = int(width * crop_fraction)
    start_h = np.random.randint(0, height - h_crop + 1)
    start_w = np.random.randint(0, width - w_crop + 1)
    image = image[start_h:start_h + h_crop, start_w:start_w + w_crop]

    # Randomly adjust brightness and contrast
    alpha = 1.0 + np.random.uniform(-0.2, 0.2)
    beta = np.random.uniform(-20, 20)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Randomly adjust saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_scale = np.random.uniform(0.5, 1.5)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Randomly rotate the image
    angle = np.random.uniform(-10, 10)
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return image
# Iterate over each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Load the original image
        image_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(image_path)

        # Apply data augmentation to generate augmented images
        num_augmented_images = 5  # Adjust the number of augmented images as needed
        augmented_images = [apply_augmentation(original_image) for _ in range(num_augmented_images)]

        # Save augmented images to the output folder
        base_name, extension = os.path.splitext(filename)
        for i, augmented_image in enumerate(augmented_images):
            output_filename = f"{base_name}_aug_{i + 1}{extension}"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, augmented_image)

print("Data augmentation completed.")
