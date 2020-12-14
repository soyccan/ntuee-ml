"""# Model

定義我們的 baseline autoencoder。
"""
import torch.nn as nn


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            # input = WxHxC = 32x32x3
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 16x16x64
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 8x8x128
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 4x4x256
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 2x2x256
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder = nn.Sequential(
            # 2x2x256
            nn.ConvTranspose2d(256, 128, 3, stride=1),  # +(3-1) = +2
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            # 4x4x128
            nn.ConvTranspose2d(128, 64, 5, stride=1),  # +(5-1) = +4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            # 8x8x128
            nn.ConvTranspose2d(64, 32, 9, stride=1),  # +(9-1) = +8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),

            # 16x16x32
            nn.ConvTranspose2d(32, 3, 17, stride=1),  # +(17-1) = +16
            nn.BatchNorm2d(3),
            nn.Tanh()

            # 32x32x3
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x1, x
