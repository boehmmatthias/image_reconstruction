import json
import os
from torch.utils.data import DataLoader
from dataset import get_datasets
from model import UNet
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


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
    criterion = nn.MSELoss()
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
        for inputs, targets in tqdm(dataloaders[phase], total=len(dataloaders[phase]), desc=phase):
            #inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            #targets = targets.squeeze(1)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses[phase].append(loss.item())
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
        print(f'{phase} Loss: {np.mean(losses[phase]):.4f}')
    return losses, model


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    prepare_train(config)
