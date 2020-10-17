import pandas as pd
import numpy as np
import joblib

class Dataset:
    def __init__(self):
        self.X = None
        self.X_t = None
        self.y_hat = None
        self.df = None
        self.w = None
    def normalize(self):
        mean_x = np.nanmean(X, axis=1)
        std_x = np.nanstd(X, axis=1)
        std_x[std_x == 0] = 1
        self.X = (self.X - mean_x[:, np.newaxis]) / std_x[:, np.newaxis]

trainset = Dataset()

trainset.df = pd.read_csv('train_datas_0.csv', dtype='float', na_values='-')
trainset.df1 = pd.read_csv('train_datas_1.csv').apply(pd.to_numeric, errors='coerce')

DIM = 15*9 + 3 # with 2nd-order term, hour, day and bias

def fill_nan_with_mean():
    trainset.df1.dropna(how='all')
    for s in (trainset.df, trainset.df1):
        s[s.columns] = s[s.columns].fillna(s[s.columns].mean())
fill_nan_with_mean()

def preprocess_training_data():
    d = DIM
    n = len(trainset.df) - 9 #+ len(trainset.df1) - 9
    X = np.zeros((d, n))
    y_hat = np.zeros(n)

    # extract feature
    hr = 0
    day = 0
    for i in range(len(trainset.df)-9):
        X[0:15*9, i] = trainset.df.iloc[i:i+9].to_numpy().flatten()
        #X[15*9:15*9*2, i] = X[0:15*9, i] ** 2
        X[15*9, i] = hr
        X[15*9+1, i] = day
        X[15*9+2, i] = 1 # bias
        hr = (hr + 1) % 24
        day = (day+1) % 365
        y_hat[i] = trainset.df.iloc[i+9]['PM2.5']
#    hr = 0
#     off = len(trainset.df)-9
#     for i in range(len(trainset.df1)-9):
#         X[0:15*9, off+i] = trainset.df1.iloc[i:i+9].to_numpy().flatten()
#         X[15*9:15*9*2, off+i] = X[0:15*9, off+i] ** 2
#         X[15*9*2, off+i] = hr
#         hr = (hr + 1) % 24
#         y_hat[i] = trainset.df1.iloc[i+9]['PM2.5']

    trainset.X = X
    trainset.X_t = X.T
    trainset.y_hat = y_hat
    #trainset.normalize()

preprocess_training_data()

def train(X, y_hat, lr, reg, num_iter):
    # training
    X_t = X.T
    d, n = X.shape
    w = np.zeros(d)
    sum_grad_sq = 0 # adagrad
    loss = np.zeros(num_iter)
    for i in range(num_iter):
        # L(w) = || X^T - y^ ||^2 + reg * ||w||^2
        # ∇L(w) = 2 * X * (X^T * w - y^) + 2 * reg * w
        y = np.dot(X_t, w)
        loss[i] = np.inner(y - y_hat, y - y_hat) + reg * np.linalg.norm(w, 1)
        grad = np.dot(X, y - y_hat) + reg * w
        sum_grad_sq += np.inner(grad, grad)
        w = w - lr / np.sqrt(sum_grad_sq) * grad
        if i == num_iter-1:
            print('i',i)
            #print('std_x',std_x)
            #print('rr',rr)
            #print('Xt',X_t)
            print('y', y)
            print('y_hat', y_hat)
            #print('grad', grad)
            print('w', w)
            print('loss', loss[i])
            print()
    plt.plot(loss)
    return w

def validate(X, y_hat, w):
    # validate
    X_t = X.T
    y = np.dot(X_t, w)
    return np.inner(y - y_hat, y - y_hat)

def run():
    X = trainset.X
    y_hat = trainset.y_hat
    d, n = X.shape
    idx_validate = pd.Series([False] * n)
    idx_validate[0:1000] = True
    
    lr = 1e-1 # learning rate
    reg = 1e-3 # regularization term, lasso or ridge
    
    for lr in (0.1, 0.05):
        for reg in (0,):
            print('lr',lr,'reg',reg)
            trainset.w = train(X[:, ~idx_validate], y_hat[~idx_validate], lr, reg, 10000)
            loss = validate(X[:, idx_validate], y_hat[idx_validate], trainset.w)
            print('validate loss', loss)
            print()
run()

joblib.dump(trainset.w, 'model-main.pkl')
