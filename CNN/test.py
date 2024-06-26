import os
import json
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from CNN.dataset import ImageDataset
import wandb

from CNN.losses import CustomWeightedMSELoss


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
    criterion = CustomWeightedMSELoss(0.7, 0.3)
    criterion.to(device)
    counter = 0
    for x, y, crop_starts, crop_ends in tqdm(test_loader, total=len(test_loader)):
        with torch.no_grad():
            output = model(x.to(device))
            cropped_predictions = torch.zeros(size=(len(crop_starts), 3, 30, 20)).to(device)
            cropped_targets = torch.zeros(size=(len(crop_starts), 3, 30, 20)).to(device)

            for i, _ in enumerate(crop_starts):
                result_path = 'results'
                # gt_image = transforms.ToPILImage()(targets[i].squeeze(0))
                # gt_image.save(os.path.join(result_path, f'{i}_gt.png'))
                start_x, start_y = crop_starts[i]
                end_x, end_y = crop_ends[i]
                target_copy = output.clone().detach()
                for channel in range(config['out_channels']):
                    cropped_output = output[i, channel]
                    cropped_target = target_copy[i, channel]
                    cropped_predictions[i, channel, :30, :20] = cropped_output[start_y:end_y, start_x:end_x]
                    cropped_targets[i, channel, :30, :20] = cropped_target[start_y:end_y, start_x:end_x]

            loss = criterion(y.to(device), output.to(device), cropped_targets.to(device), cropped_predictions.to(device))
            # loss = criterion(outputs, targets)
            wandb.log({f'test_loss': loss.item()})

            output = output.squeeze(0)
            # convert to image and save
            input_im = transforms.ToPILImage()(x.cpu().squeeze(0))
            input_im.save(os.path.join(result_path, f'{counter}_input.png'))
            output = transforms.ToPILImage()(output.cpu().squeeze(0))
            output.save(os.path.join(result_path, f'{counter}_result.png'))
            gt_image = transforms.ToPILImage()(y.squeeze(0))
            gt_image.save(os.path.join(result_path, f'{counter}_gt.png'))

            counter += 1


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    wandb.init(
        # set the wandb project where this run will be logged
        project="nn_image_reconstruction",

        # track hyperparameters and run metadata
        config=config
    )

    test_model(config)
    wandb.finish()
