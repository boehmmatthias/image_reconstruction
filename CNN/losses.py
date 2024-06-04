from torchvision import models
import torch.nn.functional as F
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=True).features
        self.feature_extractor = nn.Sequential(*list(vgg[:18])).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return F.mse_loss(pred_features, target_features)


import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weight_scale=10.0):
        super(WeightedMSELoss, self).__init__()
        self.weight_scale = weight_scale

    def forward(self, input, target):
        error = input - target
        weight = torch.exp(-self.weight_scale * torch.abs(error))
        weighted_mse = weight * (error ** 2)
        return torch.mean(weighted_mse)


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output_full, target_full, output_crop, target_crop):
        loss1 = self.mse_loss(output_full, target_full)
        loss2 = self.mse_loss(output_crop, target_crop)
        total_loss = loss1 + loss2
        return total_loss


class CustomWeightedMSELoss(nn.Module):
    def __init__(self, weight_full=0.5, weight_cropped=0.5):
        super(CustomWeightedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.weight_full = weight_full
        self.weight_cropped = weight_cropped

    def forward(self, output1, target1, output2, target2):
        # Calculate MSE loss for the full-size tensors
        loss_full = self.mse_loss(output1, target1)

        # Calculate MSE loss for the cropped tensors
        loss_cropped = self.mse_loss(output2, target2)

        # Combine the losses
        total_loss = self.weight_full * loss_full + self.weight_cropped * loss_cropped

        return total_loss
