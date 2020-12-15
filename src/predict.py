"""接著我們使用訓練好的 model，來預測 testing data 的類別。

由於 testing data 與 training data 一樣，因此我們使用同樣的 dataset 來實作 dataloader。與 training 不同的地方在於 shuffle 這個參數值在這邊是 False。

準備好 model 與 dataloader，我們就可以進行預測了。

我們只需要 encoder 的結果（latents），利用 latents 進行 clustering 之後，就可以分類了。
"""

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from common import *


def predict(X_embedded):
    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred


def invert(pred):
    return np.abs(1 - pred)


def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')


same_seeds(0)

# load model
# model = torch.load('../model/0.82888.pth')  #, map_location='cpu')
model = torch.load(MODEL_PATH)
model.eval()

# 準備 data
trainX = np.load('trainX.npy')
trainY = np.load('trainY.npy')

# 預測答案
latents = inference(X=trainX, model=model)
X_embedded = reduce_dim(latents)
pred = predict(X_embedded)

# Problem c (作圖) 將 train data 的降維結果 (embedding) 與他們對應的 label 畫出來。
acc_latent = cal_acc(trainY, pred)
print('The clustering accuracy is:', acc_latent)
print('The clustering result:')
plot_scatter(X_embedded, trainY, savefig='clusters.png')

# 將預測結果存檔，上傳 kaggle
save_prediction(pred, 'prediction.csv')

# 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
# 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
save_prediction(invert(pred), 'prediction_invert.csv')
