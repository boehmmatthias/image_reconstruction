import os
import numpy as np
from PIL import Image, ImageFilter
import random
from tqdm import tqdm


def apply_gaussian_blur_to_rectangle(image, radius=15):
    width, height = image.size
    rect_width = 20
    rect_height = 30

    x1 = random.randint(0, width - rect_width)
    y1 = random.randint(0, height - rect_height)
    x2 = x1 + rect_width
    y2 = y1 + rect_height

    # Crop the rectangle
    rect = image.crop((x1, y1, x2, y2))

    # Apply Gaussian blur to the rectangle
    rect = rect.filter(ImageFilter.GaussianBlur(radius=radius))

    # Paste the blurred rectangle back to the image
    image.paste(rect, (x1, y1))

    return image, (x1, y1), (x2, y2)


def process_images(input_directory, output_directory, radius=15):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_directory, filename)
            image = Image.open(image_path).convert('RGB')

            blurred_image, crop_start, crop_size = apply_gaussian_blur_to_rectangle(image, radius)
            output_path = os.path.join(output_directory, filename)
            blurred_image.save(output_path)

if __name__ == "__main__":
    input_dir = '../data/128x128_images'
    output_dir = '../data/128x128_images_blurred'
    process_images(input_dir, output_dir)
