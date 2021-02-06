import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

def load_training_data(path='training_label.txt'):
    # TODO: can ' t ==> (can) ('t) rather than (can) (') (t)
    # 把 training 時需要的 data 讀進來
    # 如果是 'training_label.txt'，需要讀取 label，如果是 'training_nolabel.txt'，不需要讀取 label
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        print(lines[:5])
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def load_testing_data(path='testing_data.txt'):
    # 把 testing 時需要的 data 讀進來
    with open(path, 'r') as f:
        lines = f.readlines()

        # x=[line for line in lines]
        # print(len(x))
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        # print(len(X))
        X = [sen.split(' ') for sen in X]
        # print(len(X))
    return X


# xtest=load_testing_data()
# print(xtest[:10])
# input('stop')

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs >=0.5] = 1 # 大於等於 0.5 為有惡意
    outputs[outputs < 0.5] = 0 # 小於 0.5 為無惡意
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct
