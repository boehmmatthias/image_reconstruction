import os
import json
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from CNN.dataset import ImageDataset


def test_model(config):
    result_path = config['result_path']
    # check if result path exist. if not, create them
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # Check for MPS availability
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print('MPS device is available and will be used.')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'MPS device is not available. Using {device} instead.')

    model_path = Path(config['model_path']) / config['best_model']
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    num_samples = int(config['num_samples'])
    indices = np.arange(num_samples)
    test_set_size = int(np.ceil(num_samples * float(config['test_ratio'])))
    test_set = ImageDataset(config['data_path'], indices[:test_set_size])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    counter = 0
    for x, y in tqdm(test_loader, total=len(test_loader)):
        with torch.no_grad():
            output = model(x.to(device))
            output = output.squeeze(0)
            # convert to image and save
            input_im = transforms.ToPILImage()(x.cpu().squeeze(0))
            input_im.save(os.path.join(result_path, f'{counter}_input_.png'))
            output = transforms.ToPILImage()(output.cpu().squeeze(0))
            output.save(os.path.join(result_path, f'{counter}_result_.png'))
            gt_image = transforms.ToPILImage()(y.squeeze(0))
            gt_image.save(os.path.join(result_path, f'{counter}_gt.png'))

            counter += 1


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    test_model(config)
