# -*- coding: utf-8 -*-
"""HW5

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1selHC_KXRwEL_rLWTV9fW2m3yHWE47bQ

This is the tutorial of **Image Clustering**
<br>
If you want to skip the **training** phase, please refer to the **clustering** section directly.
<br>
**Training** required sections:  Prepare Training Data, Model, Training
<br>
**Clustering** required sections: Prepare Training Data, Model, Dimension Reduction & Clustering

同學們也可以利用提供的wget指令下載訓練資料，並自行mount到雲端資料夾上，如作業一所示。這邊就不再贅述<br>
作業的第一部分是要訓練一個autoencoder以抽取好的圖片表徵，第二部分則是將抽出來的表徵降維到二維，以便我們利用分群的方法獲得我們的答案<br>

若有任何問題，歡迎來信至助教信箱 ml2020fall@gmail.com

# Download Dataset
"""

# !gdown --id '1-BjiBb9PxYndTxOrWeDqUwk6v4TtKf7I' --output trainX.npy
# !gdown --id '11TfVM1ESD0y-X7Zh9nouj0eY0axR2PVk' --output trainY.npy #請勿將此檔案拿來train模型
# #https://drive.google.com/file/d/1-BjiBb9PxYndTxOrWeDqUwk6v4TtKf7I/view?usp=sharing
# #https://drive.google.com/file/d/11TfVM1ESD0y-X7Zh9nouj0eY0axR2PVk/view?usp=sharing
# !mkdir checkpoints
# !ls

import os
os.makedirs('checkpoints', exist_ok=True)


"""將訓練資料讀入，並且 preprocess。
之後我們將 preprocess 完的訓練資料變成我們需要的 dataset。請同學不要使用 trainY 來訓練。
"""

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from common import *

trainX = np.load('trainX.npy')
trainX = preprocess(trainX)
trainX, validX = train_test_split(trainX, test_size=0.2)
train_dataset = Image_Dataset(trainX)
valid_dataset = Image_Dataset(validX)


"""# Training

這個部分就是主要的訓練階段。
我們先將準備好的 dataset 當作參數餵給 dataloader。
將 dataloader、model、loss criterion、optimizer 都準備好之後，就可以開始訓練。
訓練完成後，我們會將 model 存下來。
"""

import torch
import torch.optim
import torch.nn
import os
import numpy as np

same_seeds(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE().to(device)
print(model)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-7)

# 準備 dataloader, model, loss criterion 和 optimizer
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_tensor = torch.tensor(validX).to(device)

n_epoch = 100
best_valid_loss = 1e9
train_history = []
# 主要的訓練過程
for epoch in range(n_epoch):
    epoch_loss = []
    model.train()
    for data in train_dataloader:
        img = data.to(device)

        output1, output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if (epoch+1) % 10 == 0:
        #     torch.save(model.state_dict(),
        #                './checkpoints/checkpoint_{}.pth'.format(epoch+1))
        epoch_loss.append(loss.item())
    epoch_loss = np.mean(epoch_loss)

    model.eval()
    valid_encoded, valid_reconstruct = model(valid_tensor)
    valid_loss = criterion(valid_reconstruct, valid_tensor).item()
    print('epoch [{}/{}], train_loss:{:.5f}, valid_loss:{:.5f}'.format(
          epoch+1, n_epoch, epoch_loss, valid_loss))
    train_history.append({'train_loss': epoch_loss, 'valid_loss': valid_loss})

    if valid_loss < best_valid_loss:
        print('Saving valid loss {}'.format(valid_loss))
        best_valid_loss = valid_loss
        torch.save(model, './checkpoints/best.pth')

