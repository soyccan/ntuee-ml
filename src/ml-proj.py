# This file is for evaluation on Kaggle
# Upload this file by:
#   kaggle kernels push -p src
#
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import os.path
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
test_imgs = next(
    os.walk(os.path.join(os.path.dirname(__file__), '../input/severstal-steel-defect-detection/test_images')))[2]
N = len(test_imgs)


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pred = pd.DataFrame({
    'ImageId': test_imgs,
    'EncodedPixels': ['1 409600']*N,
    'ClassId': np.zeros(N, dtype='int32')
})
pred.to_csv('submission.csv', index=False)
