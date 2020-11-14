import numpy as np
from tensorflow import keras
import PIL
from matplotlib import pyplot

# %% Plot feature Maps

model = keras.models.load_model('1/9.hdf5')

model_tmp = keras.Model(inputs=model.inputs,
                        outputs=[model.layers[i].output
                                 for i in range(len(model.layers))
                                 if 'conv' in model.layers[i].name])

X_train = np.load('X_train.npy').astype(np.float32)
X_test = np.load('X_test.npy').astype(np.float32)
X = np.concatenate((X_train, X_test), axis=0)
N = X.shape[0]

for i in range(100):
    img = X[i, :].reshape((1, 48, 48, 1))
    feature_maps = model_tmp.predict(img)
    for j, fmap in enumerate(feature_maps):
        for k in range(fmap.shape[3]):
            out_im = PIL.Image.fromarray(np.rint(fmap[0, :, :, k] * 255))
            out_im = out_im.convert('L')  # grayscale
            out_im.save('visualize/silency-{}-{}-{}.png'.format(i, j, k))


# %% Plot in grids
# feature maps
width = 4
height = 8
ix = 1
for i in range(height):
    for j in range(width):
        # specify subplot and turn of axis
        ax = pyplot.subplot(width, height, ix)
        ax.set_xticks([])
        ax.set_yticks([])

        im = PIL.Image.open('visualize/silency-0-3-{}.png'.format(
            i * width + j
        ))

        # plot filter channel in grayscale
        pyplot.imshow(im, cmap='gray')

        ix += 1
# show the figure
pyplot.show()
