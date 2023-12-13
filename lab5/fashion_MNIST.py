import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

''' Utworzenie unikalnego katalogu dla logów TensorBoard '''
logdir = "./logs/fashion" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_big"

''' Inicjalizacja callbacku TensorBoard do monitorowania procesu uczenia '''
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

class_names = []

''' Definicja funkcji do ładowania danych Fashion MNIST '''
def load_data(verbose=False):
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    ''' Definicja nazw klas dla zbioru danych Fashion MNIST '''
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    ''' Opcjonalne wyświetlenie pierwszego obrazu ze zbioru treningowego '''
    if verbose:
        plt.figure()
        plt.imshow(train_images[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()

    return (train_images, train_labels), (test_images, test_labels)


''' Definicja funkcji do normalizacji pikseli '''
def normalize_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm


''' Funkcja definiująca model sieci neuronowej '''
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


''' Funkcja definiująca większy model sieci neuronowej '''
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

'''
Uruchomienie funkcji:
1. Ładowanie danych MNIST
2. Normalizacja pikseli
3. Definiowanie i kompilacja większego modelu
4. Trenowanie modelu
5. Ewaluacja modelu na danych testowych
6. Wyświetlenie dokładności modelu
7. Wykonanie predykcji
'''
(train_images, train_labels), (test_images, test_labels) = load_data()

train_images, test_images = normalize_pixels(train_images, test_images)

model = define_bigger_model()

model.fit(train_images, train_labels, epochs=5, validation_split=0.2, callbacks=tensorboard_callback)

_, acc = model.evaluate(test_images, test_labels)

print(acc)

predictions = model.predict(test_images)
