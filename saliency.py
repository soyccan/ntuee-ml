import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

model = keras.models.load_model('1/9.hdf5')

_img = keras.preprocessing.image.load_img(
    'train/train/00013.jpg',
    grayscale=True,
    target_size=(48, 48))

# preprocess image to get it into the right format for the model
img = keras.preprocessing.image.img_to_array(_img)
img = img.reshape((1, *img.shape))
y_pred = model.predict(img)

images = tf.Variable(img, dtype=float)

with tf.GradientTape() as tape:
    pred = model(images, training=False)
    class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    loss = pred[0][class_idxs_sorted[0]]

grads = tape.gradient(loss, images)
dgrad_abs = tf.math.abs(grads)
dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

# normalize to range between 0 and 1
arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].imshow(_img, cmap='gray')
i = axes[1].imshow(grad_eval, cmap="jet",alpha=0.8)
fig.colorbar(i)
plt.show()

# # Find the index of the to be visualized layer above
# layer_index = utils.find_layer_idx(model, 'dense_2')
#
# # Swap softmax with linear
# model.layers[layer_index].activation = keras.activations.linear
# model = utils.apply_modifications(model)
#
# # Numbers to visualize
# indices_to_visualize = [ 0, 12, 38, 83, 112, 74, 190 ]
#
# # Visualize
# for index_to_visualize in indices_to_visualize:
#   # Get input
#   input_image = input_test[index_to_visualize]
#   # Class object
#   classes = {
#     0: 'airplane',
#     1: 'automobile',
#     2: 'bird',
#     3: 'cat',
#     4: 'deer',
#     5: 'dog',
#     6: 'frog',
#     7: 'horse',
#     8: 'ship',
#     9: 'truck'
#   }
#   input_class = np.argmax(target_test[index_to_visualize])
#   input_class_name = classes[input_class]
#   # Matplotlib preparations
#   fig, axes = plt.subplots(1, 2)
#   # Generate visualization
#   visualization = visualize_saliency(model, layer_index, filter_indices=input_class, seed_input=input_image)
#   axes[0].imshow(input_image)
#   axes[0].set_title('Original image')
#   axes[1].imshow(visualization)
#   axes[1].set_title('Saliency map')
#   fig.suptitle(f'CIFAR10 target = {input_class_name}')
#   plt.show()


# Reference:
# https://usmanr149.github.io/urmlblog/cnn/2020/05/01/Salincy-Maps.html