import pandas as pd
import numpy as np
import pickle
import sys
np.set_printoptions(precision=2, suppress=True)


class G:
    # global variables
    pass

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


G.df_train_full = pd.read_csv(sys.argv[3], dtype=np.int32)
G.df_test_full = pd.read_csv(sys.argv[5], dtype=np.int32)
G.y_train = np.array(open(sys.argv[4]).read().strip('\n').split('\n'), dtype=np.int8)



def extract(df):
    c = len(df.columns)
    n = len(df)
    d = 1 + c + 5

    X = np.zeros((n, d))
    X[:, 0] = 1  # bias
    X[:, 1:1 + c] = df.values

    # ['age', 'fnlwgt', 'hours_per_week', 'capital_gain', 'capital_loss']
    idx = [1, 2, 4, 5, 6]
    X[:, 1 + c:1 + c + 5] = X[:, idx]**2

    return X


def preprocess_train(df):
    df = df.copy()
    col_mean = df.mean()
    col_std = df.std()

    # normalize
    normcols = ['age', 'fnlwgt', 'hours_per_week']
    normcols += ['capital_gain', 'capital_loss']
    df[normcols] = (df[normcols] - df[normcols].mean()) / df[normcols].std()

    return extract(df), normcols, col_mean, col_std


G.X_train, G.normcols, G.col_mean, G.col_std = preprocess_train(
    G.df_train_full)


def preprocess_test(df, normcols, col_mean, col_std):
    df = df.copy()

    # normalize
    for cnam in normcols:
        df[cnam] = (df[cnam] - col_mean[cnam]) / col_std[cnam]

    return extract(df)


G.X_test = preprocess_test(G.df_test_full, G.normcols, G.col_mean, G.col_std)


def test(X, w):
    y = sigmoid(X @ w)
    print(y[:100])
    y = np.rint(y).astype(np.int8)
    return y

G.model = pickle.load(open('log/1.pkl', 'rb'))

G.y_test = test(G.X_test, G.model)

df_pred = pd.DataFrame({
    'id': np.arange(1,
                    len(G.X_test) + 1),
    'label': G.y_test
})
df_pred.to_csv(sys.argv[6], index=False)
df_pred['label'].values[:100]
