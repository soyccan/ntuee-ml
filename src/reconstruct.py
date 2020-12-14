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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AE().to(device)
# model.load_state_dict(torch.load(MODEL_PATH))
model = torch.load(MODEL_PATH).to(device)
model.eval()

# load data
trainX = np.load('trainX.npy')
# for i in range(100):
#     # write data as images
#     Image.fromarray(trainX[i], 'RGB').save('input/{:02}.png'.format(i))
trainX = preprocess(trainX)

# evaluate
for i in range(100):
    img = torch.tensor(trainX[i:i+1]).to(device)
    vec, img1 = model(img)
    img1 = postprocess(img1.cpu().detach().numpy())
    Image.fromarray(img1[0], 'RGB').save('reconstruct/{}.png'.format(i))