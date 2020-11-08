import numpy as np
import pandas as pd
from tensorflow import keras
import os

X_train = np.load('X_train.npy').astype(np.float32)
y_train = np.load('y_train.npy').astype(np.float32)
X_test = np.load('X_test.npy').astype(np.float32)
n_train = X_train.shape[0]
n_test = X_test.shape[0]

os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = keras.models.load_model('1/9.hdf5')

# print(model.evaluate(X_train, y_train))

y_test = model.predict_classes(X_test)
pred = pd.DataFrame({
    'id': np.arange(n_test),
    'label': y_test
})
pred.to_csv('submission.csv', index=False)
