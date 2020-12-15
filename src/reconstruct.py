"""接著我們使用訓練好的 model，來預測 testing data 的類別。

由於 testing data 與 training data 一樣，因此我們使用同樣的 dataset 來實作 dataloader。與 training 不同的地方在於 shuffle 這個參數值在這邊是 False。

準備好 model 與 dataloader，我們就可以進行預測了。

我們只需要 encoder 的結果（latents），利用 latents 進行 clustering 之後，就可以分類了。
"""

from PIL import Image
import torch
import numpy as np
from common import *

# load model
model = torch.load(MODEL_PATH).cpu()
model.eval()

# load data
trainX = np.load('trainX.npy')
# for i in range(100):
#     # write data as images
#     Image.fromarray(trainX[i], 'RGB').save('input/{:02}.png'.format(i))
trainX = preprocess(trainX)
train_dataset = Image_Dataset(trainX)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)

# evaluate
criterion = torch.nn.MSELoss()
total_loss = []
for i, img in enumerate(train_dataloader):
    vec, img1 = model(img)
    total_loss.append(criterion(img1, img).item())
print('Reconstruction Loss:', np.mean(total_loss))
for i in range(100):
    img = torch.tensor(trainX[i:i+1])
    vec, img1 = model(img)
    img1 = postprocess(img1.detach().numpy())
    Image.fromarray(img1[0], 'RGB').save('reconstruct/{}.png'.format(i))
