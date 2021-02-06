import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../work/train_history.csv')
df = df[:20]
plt.plot(df['train_loss'], label='train_loss')
plt.plot(df['val_loss'], label='valid_loss')
plt.legend()
plt.show()

plt.plot(df['val_acc'], label='valid_acc')
plt.plot(df['train_acc'], label='train_acc')
plt.legend()
plt.show()
