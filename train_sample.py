# HW3 - Face Expression Prediction
import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image

'''
    PyTorch
'''
import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

isGPU = torch.cuda.is_available()
print('PyTorch GPU device is available: {}'.format(isGPU))


# Utils
# - Load Data & Shuffle
# - Dataset
#   - [Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html)
#     - For Data Augmentation
#   - [Dataloader](https://pytorch.org/docs/stable/data.html)
def load_data(img_path='./train/train',
              label_path='./train/train/label.csv',
              shuffle=True):
    train_images = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    train_labels = pd.read_csv(label_path)
    train_labels = train_labels.iloc[:, 1:].values.tolist()
    train_datas = list(zip(train_images, train_labels))
    if shuffle:
        random.seed(2020)
        random.shuffle(train_datas)

    train_set = train_datas
    valid_set = train_datas[28500:]

    return train_set, valid_set


class dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0])
        img = self.transform(img)
        label = self.data[idx][1]
        return img, label


train_set, valid_set = load_data()
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = dataset(train_set, transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

valid_dataset = dataset(valid_set, transform)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)


# Model
# - naiveCNN
#   - [Torchvision Models](https://pytorch.org/docs/stable/torchvision/models.html)
#     - Do not use pretrained weights
class naiveCNN(nn.Module):
    def __init__(self):
        super(naiveCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 3 * 3 * 128)
        x = self.fc(x)
        return x


# Training Phase
model = naiveCNN()
if isGPU is True:
    model.cuda()

opt = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = []
    train_acc = []
    for idx, (img, label) in enumerate(train_loader):
        if isGPU:
            img = img.cuda()
            label = label[0].cuda()
        opt.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()
        opt.step()
        predict = torch.max(output, 1)[1]
        acc = np.mean((label == predict).cpu().numpy())
        train_acc.append(acc)
        train_loss.append(loss.item())
    print('----------------------------------------------------')
    print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))

    model.eval()
    with torch.no_grad():
        valid_loss = []
        valid_acc = []
        for idx, (img, label) in enumerate(valid_loader):
            if isGPU:
                img = img.cuda()
                label = label[0].cuda()
            output = model(img)
            loss = loss_fn(output, label)
            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            valid_loss.append(loss.item())
            valid_acc.append(acc)
        print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss),
                                                                        np.mean(valid_acc)))

# Save model
# ckpt_path = './model_ckpt/model_{}.pth'.format(epochs)
# torch.save(model.state_dict(), ckpt_path)
