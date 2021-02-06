# main.py
import os
import sys
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split

from rnn.common import *
from rnn.preprocess import *
from rnn.data import *
from rnn.train import *
from rnn.model import *
from w2v import *

# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 處理好各個 data 的路徑
train_with_label = sys.argv[1]
train_no_label = sys.argv[2]

w2v_path = 'model/w2v-dim300.model' # 處理 word to vec model 的路徑

# 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
sen_len = 45
fix_embedding = True  # fix embedding during training
batch_size = 256
epoch = 100
lr = 0.001
model_dir = 'model'  # model directory for checkpoint model

print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
train_x, y = load_training_data(train_with_label)
print('train_x', train_x[:10])
# train_x_no_label = load_training_data(train_no_label)

# 對 input 跟 labels 做預處理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

# Unlabeled data
# preprocess = Preprocess(train_x_no_label, sen_len, w2v_path=w2v_path)
# embedding = preprocess.make_embedding(load=True)
# train_x_no_label = preprocess.sentence_word2idx()


# 製作一個 model 的對象
model = LSTM_Net(embedding, embedding_dim=300, hidden_dim=64, num_layers=4,
                 dropout=0.75, fix_embedding=fix_embedding)
# model = torch.nn.DataParallel(model) # Multi-GPU
model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

# 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
# X_train, X_val, y_train, y_val = train_x[:100000], train_x[100000:], \
#                                  y[:100000], y[100000:]
X_train, X_val, y_train, y_val = train_test_split(train_x, y, test_size=0.2)
# print(X_train[:10])
# print(y_train[:10])
#input('stop')
#print(type(X_train))
#input('stop')
# 把 data 做成 dataset 供 dataloader 取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)
#####################

# 把 data 轉成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
#################

# 開始訓練
training(batch_size, epoch, lr, train_loader, val_loader, model, device)
