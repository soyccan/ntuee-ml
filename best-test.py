import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.svm
import joblib
import sys
np.set_printoptions(threshold=10000, suppress=True, precision=2, linewidth=80)

class Dataset:
    pass

testset = Dataset()
testset.full_df = pd.read_csv(sys.argv[1], dtype='float')

def preprocess_testing_data(sel_cols=['PM2.5']):
    testset.df = testset.full_df.copy()
    
    for cnam in testset.df.columns:
        # fill NaN with mean
        testset.df[cnam].fillna(testset.df[cnam].mean(), inplace=True)

    c = len(sel_cols)
    d = 1 + 9*c + 9*c + 9*c #c*9*9 #+ 9*c*c
    n = (len(testset.df) + 8) // 9
    X = np.zeros((n, d))
    X[:, 0] = 1 # bias

    # extract feature
    for i in range(n):
        X[i, 1:1+9*c] = testset.df.iloc[i*9:(i+1)*9][sel_cols].values.flatten()

    # cross product
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


#     sel_cols = testset.sel_cols
#     d = np.count_nonzero(sel_cols)
#     X = X[:, sel_cols]

    testset.X = X

preprocess_testing_data()#[x for x in testset.full_df.columns if x not in exclude_columns])

def test():
    n, d = testset.X.shape
    
    testset.net = joblib.load('model-best.pkl')
    
    y = testset.net.predict(testset.X)
    testset.y = y

test()

pred_df = pd.DataFrame({
    'id': ['id_' + str(i) for i in range(500)],
    'value': testset.y
})
pred_df.to_csv(sys.argv[2], index=False)

print(pred_df.values)

