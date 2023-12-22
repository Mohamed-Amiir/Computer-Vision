import os
import cv2
import numpy as np



def apply_augmentation(image, crop_fraction=None, brightness_alpha=None, brightness_beta=None, saturation_scale=None, rotate_angle=None, blur_prob=None, blur_kernel_size=None):
    augmented_images = []
    augmented_images.append(image)

    # Crop the image
    if crop_fraction is not None:
        height, width = image.shape[:2]
        h_crop_range = int(height * crop_fraction)
        h_crop = min(height, max(1, h_crop_range))
        w_crop_range = int(width * crop_fraction)
        w_crop = min(width, max(1, w_crop_range))
        start_h = np.random.randint(0, height - h_crop + 1)
        start_w = np.random.randint(0, width - w_crop + 1)
        cropped_image = image[start_h:start_h + h_crop, start_w:start_w + w_crop]
        augmented_images.append(cropped_image)

    # Adjust brightness and contrast
    if brightness_alpha is not None and brightness_beta is not None:
        brightness_contrast_adjusted = cv2.convertScaleAbs(image, alpha=brightness_alpha, beta=brightness_beta)
        augmented_images.append(brightness_contrast_adjusted)

    # Adjust saturation
    if saturation_scale is not None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        saturation_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmented_images.append(saturation_adjusted)

    # Rotate the image
    if rotate_angle is not None:
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotate_angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        augmented_images.append(rotated_image)

    # Apply Gaussian blur
    if blur_prob is not None and blur_kernel_size is not None:
        if np.random.rand() < blur_prob:
            blur_kernel_size = blur_kernel_size * 2 + 1  # Ensure odd kernel size
            blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
            augmented_images.append(blurred_image)

    return augmented_images


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
                    augmented_images = apply_augmentation(original_image,
                                      crop_fraction=0.7,
                                      brightness_alpha=1.2,
                                      brightness_beta=10,
                                      saturation_scale=1.5,
                                      rotate_angle=30,
                                      blur_prob=1.2,
                                      blur_kernel_size=3)

                    # Save augmented images to the "augmented_train" folder
                    base_name, extension = os.path.splitext(filename)
                    for i, augmented_image in enumerate(augmented_images):
                        output_filename = f"{base_name}_aug_{i + 1}{extension}"
                        output_path = os.path.join(augmented_train_folder_path, output_filename)
                        cv2.imwrite(output_path, augmented_image)


# Process augmented images as needed

print("Data augmentation completed.")
