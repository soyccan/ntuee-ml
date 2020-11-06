import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *

import os.path


class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_acc"]
        if val_acc >= self.threshold:
            self.model.stop_training = True


# TODO: !!!set to false before submitting!!!
USE_SAVED_MODEL = True

if USE_SAVED_MODEL:
    X_train = np.load('X_train.npy').astype(np.float32)
    y_train = np.load('y_train.npy').astype(np.float32)
    X_test = np.load('X_test.npy').astype(np.float32)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
else:
    n_train = len(os.listdir('train/train')) - 1
    n_test = len(os.listdir('test/test'))

    label = pd.read_csv('train/train/label.csv')
    y_train = pd.get_dummies(label['label'], columns=['label'])  # one-hot
    X_train = np.array([
        np.array(Image.open('train/train/{:05}.jpg'.format(i)))
        for i in range(n_train)]
    ).reshape((-1, 48, 48, 1))
    X_test = np.array([
        np.array(Image.open('test/test/{:05}.jpg'.format(i)))
        for i in range(n_test)]
    ).reshape((-1, 48, 48, 1))

    # np.save('X_train.npy', X_train)
    # np.save('X_test.npy', X_test)
    # np.save('y_train.npy', y_train)

print('y_train', y_train.shape, '\n', y_train)
print('X_train', X_train.shape)
print('X_test', X_test.shape)


# set GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(4)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], False)  # grow as needed
        tf.config.set_soft_device_placement(False)  # don't use CPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=2000)  # set < 2G to survive in CSIE
            ]
        )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


old_model_path = None

if old_model_path:
    model = keras.models.load_model(old_model_path)
else:
    model = keras.Sequential()
    model.add(Input(shape=(48, 48, 1)))

    # 1st Convolution Layer
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # 4th Convolution layer
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256, use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Fully connected layer 2nd layer
    model.add(Dense(512, use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(7, use_bias=True, activation='softmax'))

    model.summary()
    model.compile(
        optimizer=Adam(
            learning_rate=1e-3,
            beta_1=0.9,
            beta_2=0.99,
            # rho=0.9,
        ),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

train_history = model.fit(
    x=X_train,
    y=y_train,
    epochs=1000,
    batch_size=64,
    validation_split=0.1,
    shuffle=True,
    verbose=1,
    # use_multiprocessing=True,
    # workers=24,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            "1/9.hdf5",
            monitor='val_categorical_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max',
            save_freq='epoch'),
        # keras.callbacks.EarlyStopping(
        #     monitor="val_accuracy",
        #     baseline=0.589,
        #     min_delta=0.01,
        #     restore_best_weights=False,
        # )
    ])
