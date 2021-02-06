# main.py
import os
import sys
import torch

from rnn.preprocess import *
from rnn.test import *
from w2v import *

# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 處理好各個 data 的路徑
testing_data = sys.argv[1]
w2v_path = 'model/w2v-dim300.model' # 處理 word to vec model 的路徑

# 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
sen_len = 45
batch_size = 128

print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
test_x = load_testing_data(testing_data)

# 對 input 跟 labels 做預處理
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()

# 製作一個 model 的對象
model = torch.load('model/word2vec-0.82560.model')
model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

# 把 data 轉成 batch of tensors
test_loader = torch.utils.data.DataLoader(dataset = test_x,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

pred = testing(batch_size, test_loader, model, device)
pd.DataFrame({
    'id': np.arange(len(pred)),
    'label': pred
}).to_csv(sys.argv[2], index=False)
