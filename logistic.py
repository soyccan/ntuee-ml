import pandas as pd
import numpy as np
import joblib
np.set_printoptions(precision=2, suppress=True)


class G:
    # global variables
    pass


class LogisticRegression:
    def __init__(self, X, y, b1=0.99, b2=0.999):
        self.n = self.d = 0
        self.b1 = b1
        self.b2 = b2

        self.w = [np.zeros(self.d) for _ in range(4)]
        self.m = [np.zeros(self.d) for _ in range(4)]
        self.v = [0, 0, 0, 0]
        self.step_ctr = 0

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['X']
        del state['XT']
        del state['Xval']
        del state['y']
        del state['yval']
        return state

    def setdata(self, X, y):
        self.n, self.d = X.shape

        self.X = [X[~mask[i], :] for i in range(4)]
        self.XT = [self.X[i].T for i in range(len(self.X))]
        self.Xval = [X[mask[i], :] for i in range(4)]
        self.y = [y[~mask[i]] for i in range(4)]
        self.yval = [y[mask[i]] for i in range(4)]

        # 3-fold cross validation sets, 0 is full
        ncv = (self.n + 2) // 3
        mask = [np.zeros(self.n, dtype=np.bool_) for _ in range(4)]
        for i in range(1, 4):
            mask[i][(i - 1) * ncv:i * ncv] = True

    def _step(self, i, steps):
        for j in range(steps):
            fx = sigmoid(self.X[i] @ self.w[i])
            grad = self.XT[i] @ (fx - self.y[i])
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.sum(grad**2)
            self.w[i] = self.w[i] - self.m[i] / (1 - self.b1) / np.sqrt(
                self.v[i] / (1 - self.b2))
        loss = -self.y[i] @ np.log(fx) - (1 - self.y[i]) @ np.log(1 - fx)
        return loss

    def step(self, steps=1, log=True):
        train_loss = [self._step(i, steps) for i in range(4)]
        val_accur = []
        for i in range(4):
            y_pred = np.rint(sigmoid(self.Xval[i] @ self.w[i])).astype(np.int8)
            val_accur.append(
                np.count_nonzero(y_pred == self.yval[i]) /
                (len(self.yval[i]) + 1e-10))

        if log:
            print('train loss', np.mean(train_loss), train_loss)
            print('validation accuracy', np.mean(val_accur[1:]), val_accur)
            print(self.w[0])
            print()

        self.step_ctr += steps
        self.train_loss = train_loss
        self.val_accur = val_accur


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


G.df_train_full = pd.read_csv('X_train', dtype=np.int32)
G.df_test_full = pd.read_csv('X_test', dtype=np.int32)
G.y_train = np.array(open('Y_train').read().strip('\n').split('\n'),
                     dtype=np.int8)


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

G.model = joblib.load('1/49.pkl')


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


G.y_test = test(G.X_test, G.model.w[0])

df_pred = pd.DataFrame({
    'id': np.arange(1,
                    len(G.X_test) + 1),
    'label': G.y_test
})
df_pred.to_csv('submission.csv', index=False)
df_pred['label'].values[:100]
