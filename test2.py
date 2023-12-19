import os
import cv2
import numpy as np



def apply_augmentation(image):
    augmented_images = []
    augmented_images.append(image)
    # Randomly crop the image
    crop_fraction = np.random.uniform(0.60, 1.0) 
    height, width = image.shape[:2]
    h_crop_range = int(height * crop_fraction)
    h_crop = min(height, max(1, h_crop_range))
    w_crop_range = int(width * crop_fraction)
    w_crop = min(width, max(1, w_crop_range))
    start_h = np.random.randint(0, height - h_crop + 1)
    start_w = np.random.randint(0, width - w_crop + 1)
    cropped_image = image[start_h:start_h + h_crop, start_w:start_w + w_crop]
    augmented_images.append(cropped_image)

    # Randomly adjust brightness and contrast with increased values
    alpha = 1.0 + np.random.uniform(-0.5, 0.5)  # Increase by 30%
    beta = np.random.uniform(-15, 15)  # Increase by 30%
    brightness_contrast_adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augmented_images.append(brightness_contrast_adjusted)

    # Randomly adjust saturation with increased values
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_scale = np.random.uniform(0.7, 1.3)  # Increase by 30%
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    saturation_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(saturation_adjusted)

    # Randomly rotate the image with increased values (50% increase)
    angle = np.random.uniform(-3, 20)  # Increase by 50%
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    augmented_images.append(rotated_image)

    # Apply Gaussian blur with increased values
    if np.random.rand() > 0.5:
        blur_kernel_size = int(np.random.uniform(1, 5)) * 2 + 1  # Use odd kernel size
        blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
        augmented_images.append(blurred_image)

    return augmented_images

# def apply_augmentation(image):
#     # Randomly crop the image
#     crop_fraction = np.random.uniform(0.95, 1.0)  # Slightly smaller crop
#     height, width = image.shape[:2]
#     h_crop = int(height * crop_fraction)
#     w_crop = int(width * crop_fraction)
#     start_h = np.random.randint(0, height - h_crop + 1)
#     start_w = np.random.randint(0, width - w_crop + 1)
#     image = image[start_h:start_h + h_crop, start_w:start_w + w_crop]

#     # Randomly adjust brightness and contrast
#     alpha = 1.0 + np.random.uniform(-0.05, 0.05)  # Smaller adjustment
#     beta = np.random.uniform(-5, 5)
#     image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

#     # Randomly adjust saturation
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     saturation_scale = np.random.uniform(0.9, 1.1)  # Restrict saturation adjustment
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
#     image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#     # Randomly rotate the image
#     angle = np.random.uniform(-2, 2)  # Smaller rotation
#     rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
#     image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

#     # Apply Gaussian blur
#     if np.random.rand() > 0.5:
#         blur_kernel_size = int(np.random.uniform(1, 3)) * 2 + 1  # Use odd kernel size
#         image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

#     return image




# Root directory containing the categories
root_directory = "C:\\Users\\lenovo\\Desktop\\My-Github\\Computer-Vision\\Product Classification"

# Iterate over each category folder (assuming category folders are numbered from 1 to 20)
for category_folder in range(1, 21):
    category_folder_path = os.path.join(root_directory, str(category_folder))

    # Check if the category folder exists
    if os.path.exists(category_folder_path):
        # Path to the "Train" folder of the current category
        train_folder_path = os.path.join(category_folder_path, "Train")

        # Check if the "Train" folder exists for the current category
        if os.path.exists(train_folder_path):
            # Create the "augmented_train" folder within each category if it doesn't exist
            # augmented_train_folder_path =train_folder_path 
            augmented_train_folder_path = os.path.join(category_folder_path, "augmented_train")
            if not os.path.exists(augmented_train_folder_path):
                os.makedirs(augmented_train_folder_path)

            # Iterate over each image in the "Train" folder
            for filename in os.listdir(train_folder_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    # Load the original image
                    image_path = os.path.join(train_folder_path, filename)
                    original_image = cv2.imread(image_path)

                    # Apply data augmentation to generate augmented images
                    num_augmented_images = 1  # Adjust the number of augmented images as needed
                    augmented_images = apply_augmentation(original_image)

                    # Save augmented images to the "augmented_train" folder
                    base_name, extension = os.path.splitext(filename)
                    for i, augmented_image in enumerate(augmented_images):
                        output_filename = f"{base_name}_aug_{i + 1}{extension}"
                        output_path = os.path.join(augmented_train_folder_path, output_filename)
                        cv2.imwrite(output_path, augmented_image)


# # Root directory containing the categories
# root_directory = "C:\\Users\\lenovo\\Desktop\\My-Github\\Computer-Vision\\Product Classification"

# # Iterate over each category folder (assuming category folders are numbered from 1 to 20)
# for category_folder in range(1, 21):
#     category_folder_path = os.path.join(root_directory, str(category_folder))

#     # Check if the category folder exists
#     if os.path.exists(category_folder_path):
#         # Path to the "Train" folder of the current category
#         train_folder_path = os.path.join(category_folder_path, "Train")

#         # Check if the "Train" folder exists for the current category
#         if os.path.exists(train_folder_path):
#             # Create the "augmented_train" folder within each category if it doesn't exist
#             augmented_train_folder_path = train_folder_path
#             # augmented_train_folder_path = os.path.join(category_folder_path, "augmented_train")
#             # if not os.path.exists(augmented_train_folder_path):
#             #     os.makedirs(augmented_train_folder_path)

#             # Iterate over each image in the "Train" folder
#             for filename in os.listdir(train_folder_path):
#                 if filename.endswith(('.jpg', '.jpeg', '.png')):
#                     # Load the original image
#                     image_path = os.path.join(train_folder_path, filename)
#                     original_image = cv2.imread(image_path)

#                     # Apply data augmentation to generate augmented images
#                     num_augmented_images = 1  # Adjust the number of augmented images as needed
#                     augmented_images = [apply_augmentation(original_image) for _ in range(num_augmented_images)]

#                     # Save augmented images to the "augmented_train" folder
#                     base_name, extension = os.path.splitext(filename)
#                     for i, augmented_image in enumerate(augmented_images):
#                         output_filename = f"{base_name}_aug_{i + 1}{extension}"
#                         output_path = os.path.join(augmented_train_folder_path, output_filename)
#                         cv2.imwrite(output_path, augmented_image)

print("Data augmentation completed.")
