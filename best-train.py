import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.svm
import joblib
np.set_printoptions(threshold=10000, suppress=True, precision=2, linewidth=80)

class Dataset:
    pass

trainset = Dataset()
trainset.full_df0 = pd.read_csv('train_datas_0.csv', dtype='float', na_values='-',
                                skiprows=[61-2, 86-2])
trainset.full_df1 = pd.read_csv('train_datas_1.csv', dtype='float', na_values='-', 
                                skiprows=np.arange(2162-2, 2208-2))


exclude_columns = []

def preprocess_training_data(sel_cols=['PM2.5']):
    trainset.df = pd.concat([trainset.full_df0, trainset.full_df1])
    
    print('correct dataframe')
    trainset.df.dropna(how='all', inplace=True)
    trainset.df.drop(trainset.df.index[(trainset.df == 0).all(axis=1)], inplace=True)
    for cnam in trainset.df.columns:
        # fill NaN with mean
        trainset.df[cnam].fillna(trainset.df[cnam].mean(), inplace=True)

    trainset.df.reindex()

    print('extract feature')
    c = len(sel_cols)
    d = 1 + 9*c + 9*c + 9*c #+ c*9*9 #+ 9*c*c
    n = len(trainset.df) - 9

    X = np.zeros((n, d))
    y_hat = np.empty(n)

    print('sliding window')
    W = np.arange(9)[None, :] + np.arange(n)[:, None]
    X[:, 0] = 1 # bias
    X[:, 1:1+9*c] = trainset.df[sel_cols].values[W].reshape((n,9*c))
    y_hat = trainset.df['PM2.5'].values[np.arange(9, n+9)]

    print('add cross product of features')
#     for j in range(9):
#         st = 1 + 9*c + j*c*c
#         en = 1 + 9*c + (j+1)*c*c
#         v = np.arange(j*c, (j+1)*c)
#         X[:, st:en] = (X[:, v[:, None]] * X[:, v[None, :]]).reshape((n, c*c))
#     for j in range(c):
#         st = 1 + 9*c + j*9*9
#         en = 1 + 9*c + (j+1)*9*9
#         v = np.arange(9) * c + j
#         X[:, st:en] = (X[:, v[:, None]] * X[:, v[None, :]]).reshape((n, 9*9))
    X[:, 1+9*c:1+9*c*2] = X[:, 1:1+9*c] ** 2
    X[:, 1+9*c*2:1+9*c*3] = X[:, 1:1+9*c] ** 3

#     print('feature selection')
#     lasso = sk.linear_model.Lasso(alpha=1e-4, max_iter=1000, fit_intercept=True, normalize=False).fit(X, y_hat)
#     model = sk.feature_selection.SelectFromModel(lasso, prefit=True)
#     X = model.transform(X)

    trainset.X = X
    trainset.y_hat = y_hat

    trainset.sel_cols = np.arange(d)
#     trainset.sel_cols = model.get_support()

preprocess_training_data()#[x for x in trainset.full_df0.columns if x not in exclude_columns])

def train(X, y_hat, model_path=None):
    n, d = X.shape
    print('n,d=',n,d)

    if model_path:
        net = joblib.load(model_path)
    else:
        net = sk.linear_model.LinearRegression(fit_intercept=True, normalize=False)

    net.fit(X, y_hat)

    print()
    print('true w and bias\t', net.coef_)
    print('true loss\t', np.mean((net.predict(X) - y_hat) ** 2))
    
    trainset.net = net

def validate():
    X = trainset.X
    y_hat = trainset.y_hat
    n, d = X.shape

    idx_validate = pd.Series([False] * n)
    idx_validate[:500] = True

    train(X[~idx_validate], y_hat[~idx_validate], model_path=None)
    score = trainset.net.score(X[idx_validate], y_hat[idx_validate])
    
    true_loss = np.mean((trainset.net.predict(X[idx_validate]) - y_hat[idx_validate]) ** 2)

    print('validate score', score)
    print('validate loss', true_loss)
    print()


joblib.dump(trainset.net, 'model-best.pkl')
