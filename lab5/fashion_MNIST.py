import datetime

import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
from keras.optimizers import Adam

logdir = "./logs/fashion" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_big"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

class_names = []


def load_data(verbose=False):
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    if verbose:
        plt.figure()
        plt.imshow(train_images[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()

    return (train_images, train_labels), (test_images, test_labels)


def normalize_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm


def define_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    opt = Adam(learning_rate=0.02)
    model.compile(opt,
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def define_bigger_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    opt = Adam(learning_rate=0.02)
    model.compile(optimizer=opt,
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


(train_images, train_labels), (test_images, test_labels) = load_data(True)

train_images, test_images = normalize_pixels(train_images, test_images)

model = define_bigger_model()

model.fit(train_images, train_labels, epochs=10, validation_split=0.2, callbacks=tensorboard_callback)

_, acc = model.evaluate(test_images, test_labels)

print(acc)
