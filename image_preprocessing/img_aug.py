import os
import cv2
import numpy as np

def augment_images_in_folder(input_folder, output_folder):
    """
    Parameters:
    - input_folder: Path to the folder containing preprocessed1.0 images.
    - output_folder: Path to the folder where augmented images will be saved.
    Returns:
    - Saves randomly augmented images (flip only or flip + rotation).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Apply flip (horizontal or vertical)
            flip_type = np.random.choice([1, 0])  # 1 = horizontal, 0 = vertical
            image = cv2.flip(image, flip_type)

            # Decide randomly whether to apply rotation
            apply_rotation = np.random.choice([True, False])

            if apply_rotation:
                angle = np.random.uniform(30, 90)  # rotate between 30 to 90 degrees
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, matrix, (w, h))

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)


# Call the function
augment_images_in_folder(input_folder='preprocessed_imgs1.0', output_folder='augmented_imgs1.0')
augment_images_in_folder(input_folder='preprocessed_imgs1.1', output_folder='augmented_imgs1.1')