from torch.utils.data import Dataset
import torch
import numpy as np
import os
from torchvision import transforms


def get_datasets(config):
    num_samples = int(config['num_samples'])
    indices = np.arange(num_samples)
    train_set_size = int(np.ceil(num_samples * float(config['train_ratio'])))
    validation_set_size = int(np.ceil(num_samples * float(config['val_ratio'])))
    test_set_size = int(np.ceil(num_samples * float(config['test_ratio'])))
    train_set = ImageDataset(config['data_path'], indices[:train_set_size])
    val_set = ImageDataset(config['data_path'], indices[train_set_size:train_set_size + validation_set_size])
    test_set = ImageDataset(config['data_path'], indices[train_set_size + test_set_size:])
    return train_set, val_set, test_set


class ImageDataset(Dataset):
    def __init__(self, data_folder: str, indices):
        self.data_folder = data_folder
        self.indices = indices
        self.sample_count = len(indices)

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        index = self.indices[idx]
        image = np.load(os.path.join(self.data_folder, f'{index}_blurred.npy'))
        target = np.load(os.path.join(self.data_folder, f'{index}.npy'))
        crop_start = np.load(os.path.join(self.data_folder, f'{index}_crop_start.npy'))
        crop_end = np.load(os.path.join(self.data_folder, f'{index}_crop_end.npy'))
        input_tensor = transforms.ToTensor()(image)
        target_tensor = transforms.ToTensor()(target)

        return input_tensor, target_tensor, crop_start, crop_end
