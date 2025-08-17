import os
import cv2
import numpy as np


def preprocess_image_v0(img, target_size):
    """
    Parameters:
    - image: Input image as a NumPy array.
    - target_size: Tuple specifying the target size for resizing (width, height).
    Returns:
    - Preprocessed image as a NumPy array.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # (0) Load to greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # (1) Resize & Padding the image
    #Get original dimensions
    h, w = img.shape[:2]
    h_n, w_n = target_size
    #Calculate the scaling factor
    scale = min(w_n / w, h_n / h)
    #Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    #Resize
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    #Get new dimensions
    h, w = img.shape[:2]
    h_n, w_n = target_size
    #Calculate padding
    delta_w = w_n - w
    delta_h = h_n - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    #Black Padding the image
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # (2) Edge-preserving + noise removal using bilateral filter
    img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=25)

    # (3) Mild sharpening (optional, not too aggressive to preserve wound textures)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    img = cv2.addWeighted(img, 1.3, blurred, -0.3, 0)

    # (4) CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    img = clahe.apply(img)

    # (5) Image Normalization to range [0-1]
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img


def preprocess_image_v1(img, target_size):
    """
    Parameters:
    - img: Input image as a NumPy array.
    - target_size: Tuple specifying the target size for resizing (width, height).

    Returns:
    - Preprocessed image as a NumPy array (float32 in range [0,1]).
    """
    # (1) Resize the image while preserving aspect ratio and padding to target size
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding to reach target size
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Use mean padding color instead of black to avoid harsh edges
    mean_color = img.mean(axis=(0, 1)).astype(int)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=mean_color.tolist())

    # (2) Edge-preserving noise removal using bilateral filter
    img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=25)

    # (3) Mild sharpening (optional, not too aggressive to preserve wound textures)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    img = cv2.addWeighted(img, 1.3, blurred, -0.3, 0)

    # (4) Normalize to [0, 1] float32
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img



def preprocess_images_in_folder(input_folder, output_folder, target_size, preprocess_mode = None):
    """
    Parameters:
    - input_folder: Path to the folder containing input images.
    - output_folder: Path to the folder where preprocessed1.0 images will be saved.
    - target_size: Tuple specifying the target size for resizing (width, height).
    Returns:
    - preprocessed_imgs folder of preprocessed1.0 images saved in it.
    """
    # (1) Create an o/p folder if it doesn't exist to save the preprocessed1.0 images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # (2) Iterate through all images in the i/p folder to preprocess them
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
            # Specify then Load the image path
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            # Preprocess the image
            preprocessed_image = preprocess_mode(image, target_size)
            # Save the preprocessed1.0 image
            output_path = os.path.join(output_folder, filename)
            # Convert to 0-255 scale for saving
            cv2.imwrite(output_path,(preprocessed_image * 255).astype(np.uint8))


# Call the function
preprocess_images_in_folder(input_folder='images', output_folder='preprocessed_imgs1.0', target_size=(640,640), preprocess_mode=preprocess_image_v0)
preprocess_images_in_folder(input_folder='images', output_folder='preprocessed_imgs1.1', target_size=(640,640), preprocess_mode=preprocess_image_v1)