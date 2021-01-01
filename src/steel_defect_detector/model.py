import torch
import torch.nn


class UnetDownBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UnetDownBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size
        self.elu = torch.nn.ELU()

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.elu(self.bn2(self.conv2(x)))
        # x = self.relu(self.bn3(self.conv3(x)))
        return x


class UnetUpBlock(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UnetUpBlock, self).__init__()
        # self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        self.up_sampling = torch.nn.ConvTranspose2d(input_channel, input_channel, 2, stride=2)
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.elu(self.bn2(self.conv2(x)))
        # x = self.relu(self.bn3(self.conv3(x)))
        return x


class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.down_block1 = UnetDownBlock(1, 8, False)
        self.down_block2 = UnetDownBlock(8, 16, True)
        self.down_block3 = UnetDownBlock(16, 32, True)
        self.down_block4 = UnetDownBlock(32, 64, True)
        self.down_block5 = UnetDownBlock(64, 128, True)

        self.mid_conv1 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.mid_conv2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.mid_conv3 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.up_block1 = UnetUpBlock(64, 128, 64)
        self.up_block2 = UnetUpBlock(32, 64, 32)
        self.up_block3 = UnetUpBlock(16, 32, 16)
        self.up_block4 = UnetUpBlock(8, 16, 8)

        self.last_conv1 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm2d(8)
        self.last_conv2 = torch.nn.Conv2d(8, 4, 1, padding=0)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.elu = torch.nn.ELU()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x5 = self.elu(self.bn1(self.mid_conv1(self.x5)))
        self.x5 = self.elu(self.bn2(self.mid_conv2(self.x5)))
        # self.x5 = self.relu(self.bn3(self.mid_conv3(self.x5)))
        x = self.up_block1(self.x4, self.x5)
        x = self.up_block2(self.x3, x)
        x = self.up_block3(self.x2, x)
        x = self.up_block4(self.x1, x)
        x = self.elu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x
