import os
from PIL import Image
from PIL.Image import Resampling
from tqdm import tqdm


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


source_folder = '../data/full_res_images'
target_folder = '../data/256x256_images'
size = 256
resize_images(source_folder, target_folder, size)