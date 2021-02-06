# w2v.py
# 這個 block 是用來訓練 word to vector 的 word embedding
# 注意！這個 block 在訓練 word to vector 時是用 cpu，可能要花到 10 分鐘以上
import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec

from rnn.util import *
from rnn.common import *

def train_word2vec(x):
    # 訓練 word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=300, window=10, min_count=2,
                              workers=24, iter=2, sg=1)
    return model

def main(train_path='training_label.txt',
         train_nolabel_path='training_nolabel.txt',
         test_path='testing_data.txt',
         model_path='w2v_all.model'):
    print("loading training data ...")
    train_x, y = load_training_data(train_path)
    train_x_no_label = load_training_data(train_nolabel_path)

    print("loading testing data ...")
    test_x = load_testing_data(test_path)

    print('training')
    model = train_word2vec(train_x + train_x_no_label + test_x)

    print("saving model ...")
    model.save(model_path)

if __name__ == '__main__':
    main(train_path='work/training_label.txt',
         train_nolabel_path='work/training_nolabel.txt',
         test_path='work/testing_data.txt',
         model_path='work/w2v_all.model')