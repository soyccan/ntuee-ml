"""# Model

定義我們的 baseline autoencoder。
"""
import torch.nn as nn


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            # input = WxHxC = 32x32x3
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 16x16x32
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 8x8x64
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 4x4x128
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 2x2x256
        )

        # latent code = 1024

        self.decoder = nn.Sequential(
            # 2x2x256
            nn.ConvTranspose2d(256, 128, 3, stride=1),  # +(3-1) = +2
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            # 4x4x128
            nn.ConvTranspose2d(128, 64, 5, stride=1),  # +(5-1) = +4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            # 8x8x64
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


class AE_plus(nn.Module):
    def __init__(self):
        super(AE_plus, self).__init__()

        self.encoder = nn.Sequential(
            # input = WxHxC = 32x32x3
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 16x16x32
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 8x8x64
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 4x4x128
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 2x2x256
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            # 1x1x512
        )
        self.linear1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # latent code = 256

        self.linear2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.decoder = nn.Sequential(
            # 1x1x512
            nn.ConvTranspose2d(512, 256, 2, stride=1),  # +(2-1) = +1
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),

            # 2x2x256
            nn.ConvTranspose2d(256, 128, 3, stride=1),  # +(3-1) = +2
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            # 4x4x128
            nn.ConvTranspose2d(128, 64, 5, stride=1),  # +(5-1) = +4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            # 8x8x64
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
        x1 = self.linear1(self.encoder(x).view(-1, 512))
        x = self.decoder(self.linear2(x1).view(-1, 512, 1, 1))
        return x1, x
