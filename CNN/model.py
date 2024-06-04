import torch
import torch.nn as nn


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpsampleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleConvolution, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True)

    def forward(self, x):
        x = self.upsample(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims=None):
        super(UNet, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for hidden_dim in hidden_dims:
            self.encoders.append(DoubleConvolution(prev_channels, hidden_dim))
            prev_channels = hidden_dim

        self.bottleneck = DoubleConvolution(hidden_dims[-1], hidden_dims[-1] * 2)

        # Decoders
        self.decoders = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()
        for hidden_dim in reversed(hidden_dims):
            self.upsample_convs.append(UpsampleConvolution(hidden_dim*2, hidden_dim))
            self.decoders.append(DoubleConvolution(hidden_dim*2, hidden_dim))

        self.conv1x1 = nn.Conv2d(hidden_dims[0], out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # Encode
        enc_features = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_features.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        enc_features = enc_features[::-1]
        for i in range(len(self.decoders)):
            x = self.upsample_convs[i](x)
            x = torch.cat([x, enc_features[i]], dim=1)
            x = self.decoders[i](x)

        # add smoothing layer
        x = self.conv1x1(x)
        x = torch.sigmoid(x)

        return x
