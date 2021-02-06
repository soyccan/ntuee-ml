import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys

import numpy as np
import pandas as pd
from tensorflow import keras
from PIL import Image

test_path = sys.argv[1]
pred_path = sys.argv[2]

n_test = len(os.listdir(test_path))
X_test = np.array([
    np.array(Image.open(os.path.join(test_path, '{:05}.jpg'.format(i))))
    for i in range(n_test)
]).reshape((-1, 48, 48, 1))

model = keras.models.load_model('1/9.hdf5')

y_test = model.predict_classes(X_test)
pred = pd.DataFrame({'id': np.arange(n_test), 'label': y_test})
pred.to_csv(pred_path, index=False)
