import sys
import pandas as pd
import numpy as np
import joblib

class Dataset:
    pass

DIM = 15*9 + 3 # with 2nd-order term, hour, day and bias

testset = Dataset()
testset.df = pd.read_csv(sys.argv[1], dtype='float', na_values='-')

def preprocess_testing_data():
    for s in (testset.df,):
        s[s.columns] = s[s.columns].fillna(s[s.columns].mean())

    d = DIM
    n = (len(testset.df) + 8) // 9
    X = np.zeros((d, n))
    X[d-1, :] = 1 # for bias

    # extract feature
    hr = 0
    for i in range(n):
        X[0:15*9, i] = testset.df.iloc[i*9:(i+1)*9].to_numpy().flatten()
        #X[15*9:15*9*2, i] = X[0:15*9, i] ** 2
        X[15*9, i] = hr
        hr = (hr +1 )%24

    testset.X = X
    testset.X_t = X.T

    #testset.normalize()

preprocess_testing_data()

def predict():
    testset.w = joblib.load('model-main.pkl')
    y_pred = np.dot(testset.X_t, testset.w).round(0).astype(int)
    return y_pred
y_pred = predict()

pred_df = pd.DataFrame({
    'id': ['id_' + str(i) for i in range(500)],
    'value': y_pred
})
pred_df.to_csv(sys.argv[2], index=False)

print(pred_df.values)

