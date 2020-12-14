"""Model path that is shared across all Python code

"""
MODEL_PATH = './checkpoints/best.pth'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + os.path.dirname(__file__) \
                           if os.environ.get('PYTHONPATH') \
                           else os.path.dirname(__file__)


"""# Prepare Training Data

定義我們的 preprocess：將圖片的數值介於 0~255 的 int 線性轉為 -1～1 的 float。
"""
def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list

def postprocess(image_list):
    """ Inverse of preprocess() """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 2, 3, 1))
    image_list = (image_list + 1) / 2 * 255.0
    image_list = image_list.astype(np.uint8)
    return image_list

from torch.utils.data import Dataset

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images


"""# Some useful functions

這邊提供一些有用的 functions。
一個是計算 model 參數量的（report 會用到），另一個是固定訓練的隨機種子（以便 reproduce）。
"""

import random
import torch


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


"""# Model

定義我們的 baseline autoeocoder。
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


"""# Dimension Reduction & Clustering"""

import numpy as np

def cal_acc(gt, pred):
    """ Computes categorization accuracy of our task.
    Args:
      gt: Ground truth labels (9000, )
      pred: Predicted labels (9000, )
    Returns:
      acc: Accuracy (0~1 scalar)
    """
    # Calculate Correct predictions
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    # 因為是 binary unsupervised clustering，因此取 max(acc, 1-acc)
    return max(acc, 1-acc)

import matplotlib.pyplot as plt

def plot_scatter(feat, label, savefig=None):
    """ Plot Scatter Image.
    Args:
      feat: the (x, y) coordinate of clustering result, shape: (9000, 2)
      label: ground truth label of image (0/1), shape: (9000,)
    Returns:
      None
    """
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c = label)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return


from torch.utils.data import DataLoader

def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        vec = vec.view(img.size()[0], -1).cpu().detach().numpy()
        if i == 0:
            latents = vec
        else:
            latents = np.concatenate((latents, vec), axis=0)
    print('Latents Shape:', latents.shape)
    return latents