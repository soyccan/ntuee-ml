import pandas as pd
import numpy as np
import sys
np.set_printoptions(precision=6, suppress=True)

class G:
    # global variables
    pass

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


G.df_train_full = pd.read_csv(sys.argv[3], dtype=np.int32)
G.df_test_full = pd.read_csv(sys.argv[5], dtype=np.int32)
G.y_train = np.array(open(sys.argv[4]).read().strip('\n').split('\n'), dtype=np.int8)

def normalize(df, means, stds):
    cols = df.columns
    df = df.copy()
    df[cols] = (df[cols] - means[cols]) / stds[cols]
    return df

def extract(df):
    c = len(df.columns)
    n = len(df)
    d = c

    X = np.zeros((n, d), dtype=np.float64)
    X[:, 0:c] = df.values

    return X

def preprocess(df_train, df_test):
    df_all = pd.concat((df_train, df_test))
    means = df_all.mean()
    stds = df_all.std()
    
    df_train = normalize(df_train, means, stds)
    df_test = normalize(df_test, means, stds)
    X_train = extract(df_train)
    X_test = extract(df_test)
    
    print('n,d', X_train.shape)

    return X_train, X_test

G.X_train, G.X_test = preprocess(G.df_train_full, G.df_test_full)


class Model:
    # continuous columns: 'age', 'fnlwgt', 'hours_per_week', 'capital_gain', 'capital_loss'
    # others are binary columns (value is 0 or 1)
#     cont_cols = np.array([0, 1, 3, 4, 5])    
    cont_cols = np.arange(106)
    bin_cols = np.setdiff1d(np.arange(106), cont_cols)

    def __init__(self, X, y):
        n, d = X.shape
        XT = X.T
        X0_cont = XT[Model.cont_cols[:, None], y == 0]
        X1_cont = XT[Model.cont_cols[:, None], y == 1]
        X0_bin = XT[Model.bin_cols[:, None], y == 0]
        X1_bin = XT[Model.bin_cols[:, None], y == 1]
        n1 = np.count_nonzero(y)
        n0 = n - n1

        mean0 = np.mean(X0_cont, axis=1)
        mean1 = np.mean(X1_cont, axis=1)
        std0 = np.std(X0_cont, axis=1)
        std1 = np.std(X1_cont, axis=1)
        cov0 = np.cov(X0_cont)
        cov1 = np.cov(X1_cont)
        cov = n0 / n * cov0 + n1 / n * cov1
        cov_inv = np.linalg.inv(cov)

        self.w = (mean0 - mean1) @ cov_inv
        self.b = -0.5 * (mean0 @ cov_inv @ mean0 - mean1 @ cov_inv @ mean1) + np.log(n0 / n1)    
        self.n0, self.n1 = n0, n1

    def predict(self, X):
        n, d = X.shape

        # px := P(x|C0)
        px = sigmoid(X @ self.w + self.b)
        
        return np.rint(1-px).astype(np.int8)

G.model = Model(G.X_train, G.y_train)
G.y_test = G.model.predict(G.X_test)
df_pred = pd.DataFrame({
    'id': np.arange(1, len(G.X_test)+1),
    'label': G.y_test
})
df_pred.to_csv(sys.argv[6], index=False)
print(df_pred['label'].values[:100])
