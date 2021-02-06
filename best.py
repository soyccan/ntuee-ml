import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.experimental import enable_hist_gradient_boosting
import sklearn.ensemble
import sklearn.metrics
import joblib
import sys
np.set_printoptions(precision=2, suppress=True)

class G:
    # global variables
    pass

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
    d = 1 + c

    X = np.zeros((n, d), dtype=np.float64)
    X[:, 0] = 1  # bias
    X[:, 1:1+c] = df.values

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

G.model = joblib.load('boost/3.pkl')

G.y_test = G.model.predict(G.X_test)
df_pred = pd.DataFrame({
    'id': np.arange(1, len(G.X_test)+1),
    'label': G.y_test
})
df_pred.to_csv(sys.argv[6], index=False)
print(df_pred['label'].values[:100])
