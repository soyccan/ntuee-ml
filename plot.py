import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import re

# %% Parse Log
acc = []
loss = []
for ln in open('1/9.log'):
    match = re.search(r'loss: ([0-9\.]+) - '
                      'categorical_accuracy: ([0-9\.]+) - '
                      'val_loss: ([0-9\.]+) - '
                      'val_categorical_accuracy: ([0-9\.]+)',
                      ln)
    if match:
        loss.append((match.group(1), match.group(3)))
        acc.append((match.group(2), match.group(4)))
np.save('1/9.loss.npy', np.array(loss, dtype='float32'))
np.save('1/9.acc.npy', np.array(acc, dtype='float32'))

# %% Plot Accuracy
L = np.load('1/9.acc.npy')
plt.plot(L[:, 0], label='acc')
plt.plot(L[:, 1], label='val_acc')
plt.legend()
plt.show()

# %% Plot Loss
L = np.load('1/9.loss.npy')
plt.plot(L[:, 0], label='loss')
plt.plot(L[:, 1], label='val_loss')
plt.legend()
plt.show()

# %% Plot Confusion Matrix
y_pred = np.load('y_train_pred.npy')
y_true = np.argmax(np.load('y_train.npy'), axis=1)  # one-hot decode

conf = pd.crosstab(y_true, y_pred)
conf = conf / conf.sum(axis=1)

fig, ax = plt.subplots()

im = ax.matshow(conf, cmap='Purples', norm=LogNorm(vmin=0.01, vmax=1))
for (i, j), z in np.ndenumerate(conf):
    if i == j:
        ax.text(j, i, '{:.1%}'.format(z), ha='center', va='center', color='white')
    else:
        ax.text(j, i, '{:.1%}'.format(z), ha='center', va='center')

label_names = 'Angry Disgust Fear Happy Sad Surprise Neutral'.split(' ')
ax.set_xticklabels(label_names)
ax.set_yticklabels(label_names)

tick_marks = np.arange(len(conf.columns))
ax.xaxis.set_tick_params(rotation=45)
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_ticks(tick_marks)
ax.yaxis.set_ticks(tick_marks)

ax.set_title('Confusion Matrix')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

fig.colorbar(im, ticks=np.arange(0, 1, 0.1), format='$%.2f$')
fig.show()