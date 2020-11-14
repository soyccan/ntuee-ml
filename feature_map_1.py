# Reference:
# https://github.com/orbxball/ML2017/blob/master/hw3/activate_filters.py
import sys
import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
tf.executing_eagerly()
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np
from PIL import Image

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # print(x.shape)
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    filter_images = []
    step = 1e-2
    for i in range(num_step):
        loss_value, grads_value = iter_func([input_image_data, 0])
        input_image_data += grads_value * step
        if i % RECORD_FREQ == 0:
            filter_images.append((input_image_data, loss_value))
            print('#{}, loss rate: {}'.format(i, loss_value))
    return filter_images

RECORD_FREQ = 10



def main():
    model_name = '1/9.hdf5'

    num_step = NUM_STEPS = 100
    nb_filter = 32

    base_dir = './'
    filter_dir = os.path.join(base_dir, 'filter_vis')
    if not os.path.exists(filter_dir):
        os.mkdir(filter_dir)
    store_path = ''

    emotion_classifier = keras.models.load_model(model_name)
    collect_layers = [l.output
                      for l in emotion_classifier.layers
                      if 'batch' in l.name]

    model_tmp = keras.models.Model(
            inputs=emotion_classifier.input,
            outputs=emotion_classifier.get_layer('conv2d_3').output)

    n_filter = model_tmp.output_shape[3]
    for i_filter in range(n_filter):
        print('filter',i_filter)

        # activation = lambda y_true, y_pred: -tf.math.reduce_mean(y_true[:,:,:,i_filter])
        # model_tmp.compile(
        #     optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.99),
        #     loss=activation,
        #
        # )

        @tf.function
        def train_step(img):
            with tf.GradientTape() as tape:
                tape.watch(img)
                conv_out = model_tmp(img)
                activation = -tf.math.reduce_mean(conv_out[:, :, :, i_filter])
            grad = tape.gradient(activation, img)
            opt.apply_gradients([(grad, img)])
            return activation

        epochs = 10000
        img = tf.Variable(np.random.random((1, 48, 48, 1)), dtype='float32')
        opt = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.99)
        for i_epoch in range(epochs):
            activation = train_step(img)
            if i_epoch % 1000 == 0:
                print('epoch',i_epoch,'activation',activation)

        img = img.numpy()
        img_max = np.max(img)
        img_min = np.min(img)
        img = (img - img_min) / (img_max - img_min + 1e-18) * 255

        img = Image.fromarray(img.reshape((48,48))).convert('L')
        img.save('featmap/3-{}.png'.format(i_filter))

    # for cnt, c in enumerate(collect_layers):
    #     filter_imgs = []
    #     for filter_idx in range(nb_filter):
    #         input_img_data = np.random.random((1, 48, 48, 1)) # random noise
    #         with tf.GradientTape() as tape:
    #             y_pred = emotion_classifier(input_img_data)
    #             target = K.mean(c[:, :, :, filter_idx])
    #             grads = normalize(tape.gradient(target, input_img)[0])
    #         iterate = K.function([input_img,
    #                               K.learning_phase()],
    #                              [target, grads])
    #
    #         ###
    #         "You need to implement it."
    #         print('==={}==='.format(filter_idx))
    #         filter_imgs.append(grad_ascent(num_step, input_img_data, iterate))
    #         ###
    #     print('Finish gradient')
    #
    #     for it in range(NUM_STEPS//RECORD_FREQ):
    #         print('In the #{}'.format(it))
    #         fig = plt.figure(figsize=(14, 8))
    #         for i in range(nb_filter):
    #             ax = fig.add_subplot(nb_filter/8, 8, i+1)
    #             raw_img = filter_imgs[i][it][0].squeeze()
    #             ax.imshow(deprocess_image(raw_img), cmap='Blues')
    #             plt.xticks(np.array([]))
    #             plt.yticks(np.array([]))
    #             plt.xlabel('{:.3f}'.format(filter_imgs[i][it][1]))
    #             plt.tight_layout()
    #         # fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
    #         img_path = os.path.join(filter_dir, '{}-{}'.format(
    #                 store_path,
    #                 emotion_classifier.layers[cnt].name))
    #         if not os.path.exists(img_path):
    #             os.mkdir(img_path)
    #         fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))

main()