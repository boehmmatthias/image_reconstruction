import json
import os
from torch.utils.data import DataLoader

from losses import PerceptualLoss, WeightedMSELoss, CustomMSELoss, CustomWeightedMSELoss
from dataset import get_datasets
from model import UNet
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
from torchvision import transforms


def prepare_train(config):
    train_set, val_set, test_set = get_datasets(config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # Check for MPS availability
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print('MPS device is available and will be used.')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'MPS device is not available. Using {device} instead.')
    model = UNet(config['in_channels'], config['out_channels'])
    model = model.to(device)
    criterion = CustomWeightedMSELoss(0.7, 0.3)
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    train(model, criterion, optimizer, dataloaders, device, config)


def train(model, criterion, optimizer, dataloaders, device, config):
    best_val_loss = np.inf
    epoch_since_best = 0
    best_model = None
    epochs = config['epochs']

    for epoch in range(epochs):
        print(f"EPOCH {epoch + 1}/{epochs}")
        losses, model = train_epoch(model, criterion, optimizer, dataloaders, device)
        # torch.save(model, os.path.join(config['model_path'], f'model_epoch_{epoch + 1}.pt'))
        if np.mean(losses['val']) < best_val_loss:
            best_val_loss = np.mean(losses['val'])
            best_model = model
            torch.save(best_model, os.path.join(config['model_path'], 'best_model.pt'))
            epoch_since_best = 0

        if epoch_since_best == 5:
            print('Training Done!')
            break


def train_epoch(model, criterion, optimizer, dataloaders, device):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        losses = {'train': [], 'val': []}
        for inputs, targets, crop_starts, crop_ends in tqdm(dataloaders[phase], total=len(dataloaders[phase]),
                                                            desc=phase):
            # inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            # targets = targets.squeeze(1)
            targets = targets.to(device)
            optimizer.zero_grad()

            loss = 0
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance
                cropped_predictions = torch.zeros(size=(len(crop_starts), 3, 30, 20)).to(device)
                cropped_targets = torch.zeros(size=(len(crop_starts), 3, 30, 20)).to(device)

                for i, _ in enumerate(crop_starts):
                    result_path = 'results'
                    #gt_image = transforms.ToPILImage()(targets[i].squeeze(0))
                    #gt_image.save(os.path.join(result_path, f'{i}_gt.png'))
                    start_x, start_y = crop_starts[i]
                    end_x, end_y = crop_ends[i]
                    target_copy = targets.clone().detach()
                    for channel in range(config['out_channels']):
                        cropped_output = outputs[i, channel]
                        cropped_target = target_copy[i, channel]
                        cropped_predictions[i, channel, :30, :20] = cropped_output[start_y:end_y, start_x:end_x]
                        cropped_targets[i, channel, :30, :20] = cropped_target[start_y:end_y, start_x:end_x]

                    #output_img_crop = transforms.ToPILImage()(cropped_predictions[i].cpu().squeeze(0))
                    #output_img_crop.save(os.path.join(result_path, f'{i}_result_crop.png'))
                    #gt_image_crop = transforms.ToPILImage()(cropped_targets[i].squeeze(0))
                    #gt_image_crop.save(os.path.join(result_path, f'{i}_gt_crop.png'))
                    #output_img = transforms.ToPILImage()(outputs[i].cpu().squeeze(0))
                    #output_img.save(os.path.join(result_path, f'{i}_result.png'))
                    #gt_image = transforms.ToPILImage()(targets[i].squeeze(0))
                    #gt_image.save(os.path.join(result_path, f'{i}_gt.png'))
                    #input_image = transforms.ToPILImage()(inputs[i].squeeze(0))
                    #input_image.save(os.path.join(result_path, f'{i}_input.png'))
                # combine targets and cropped_targets for loss calculation
                loss = criterion(targets, outputs, cropped_targets, cropped_predictions)
                # loss = criterion(outputs, targets)
                losses[phase].append(loss.item())
                wandb.log({f'{phase}_loss': loss.item()})
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
        print(f'{phase} Loss: {np.mean(losses[phase]):.4f}')
    return losses, model


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    wandb.init(
        # set the wandb project where this run will be logged
        project="nn_image_reconstruction",

        # track hyperparameters and run metadata
        config=config
    )

    prepare_train(config)
    wandb.finish()
