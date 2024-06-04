import os
from PIL import Image
from PIL.Image import Resampling
from tqdm import tqdm
import numpy as np
from add_blur_rect import apply_gaussian_blur_to_rectangle
from torchvision import transforms


def resize_images(source_folder, target_folder, size):
    """
    Resize all images in source_folder to size and save them in target_folder
    :param source_folder: string, folder containing images to resize
    :param target_folder: string, folder to save resized images
    :param size: int, size to resize images to
    """
    # Make sure the output folder exists
    os.makedirs(target_folder, exist_ok=True)
    for index, filename in tqdm(enumerate(tqdm(os.listdir(source_folder)))):
        if not filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            continue
        img = Image.open(os.path.join(source_folder, filename))
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_height = size
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = size
            new_height = int(new_width / aspect_ratio)

        # Resize the image so that the smaller side is 128 pixels
        resized_image = img.resize((new_width, new_height), Resampling.LANCZOS)

        # Center crop the resized image to 128x128
        left = (new_width - size) // 2
        top = (new_height - size) // 2
        right = left + size
        bottom = top + size

        # Crop the image to 128x128
        cropped_image = resized_image.crop((left, top, right, bottom))

        cropped_image.save(os.path.join(target_folder, f'{index}.jpg'))


def blur_images_and_save_as_npy_array(source_folder, target_folder):
    """
    Save all images in source_folder as numpy arrays in target_folder
    :param source_folder: string, folder containing images to save as numpy arrays
    :param target_folder: string, folder to save numpy arrays
    """
    # Make sure the output folder exists
    os.makedirs(target_folder, exist_ok=True)
    index = 0
    for filename in tqdm(os.listdir(source_folder)):
        if not filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            continue
        img = Image.open(os.path.join(source_folder, filename))
        img = img.convert('RGB')
        img_array = np.array(img)
        np.save(os.path.join(target_folder, f'{index}.npy'), img_array)

        blurred_image, crop_start, crop_end = apply_gaussian_blur_to_rectangle(img)
        blurred_img_array = np.array(blurred_image)
        np.save(os.path.join(target_folder, f'{index}_blurred.npy'), blurred_img_array)
        np.save(os.path.join(target_folder, f'{index}_crop_start.npy'), np.array(crop_start))
        np.save(os.path.join(target_folder, f'{index}_crop_end.npy'), np.array(crop_end))
        index += 1


source_folder = '../data/full_res_images'
target_folder = '../data/128x128_images'
size = 128
#resize_images(source_folder, target_folder, size)
blur_images_and_save_as_npy_array(target_folder, '../data/128x128_images_numpy')
