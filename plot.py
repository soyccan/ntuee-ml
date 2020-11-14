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


# %% Plot Silency Map




# %% Plot filter weights
from tensorflow import keras
from PIL import Image

model = keras.models.load_model('1/9.hdf5')

filter_list = []  # by layer
for i, layer in enumerate(model.layers):
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    # get filter weights
    filters, biases = layer.get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    print(layer.name, filters.shape)
    filter_list.append(filters)

    n_channels = filters.shape[2]
    n_filters = filters.shape[3]
    for j in range(n_channels):
        for k in range(n_filters):
            im = Image.fromarray(np.rint(filters[:, :, j, k] * 255))
            im = im.convert('L')  # grayscale
            im.save('visualize/filter-L{}-{}-{}.png'.format(i, j, k))

# %%
from matplotlib import pyplot
for i_layer, filters in enumerate(filter_list):
    print()
    print('========')
    print('layer ', i_layer)
    print(filters.shape)
    n_filters, ix = filters.shape[3], 1
    n_channels = filters.shape[2]

    # plot faster
    n_filters = 3
    n_channels = 1

    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(n_channels):
            # specify subplot and turn of axis
            ax = pyplot.subplot(n_filters, n_channels, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[:, :, j], cmap='gray')
            ix += 1

    # show the figure
    pyplot.show()


