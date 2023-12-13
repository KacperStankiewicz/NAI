"""
Autorzy
Rutkowski Marcin - s12497
Stankiewicz Kacper - s22619

ROzpoznawanie zwierząt na podstawie zdjęć
"""

from keras.datasets import cifar10
from keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot

''' Definicja funkcji do ładowania i opcjonalnego wyświetlania danych CIFAR-10 '''
def load_data(verbose=False):
    (trainX, trainY), (testX, testY) = cifar10.load_data()

    if verbose:
        print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
        print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
        for i in range(9):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(trainX[i])
        pyplot.show()

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY

''' Definicja funkcja normalizującej pikseli '''
def normalize_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm

''' Funkcja definiująca model sieci konwolucyjnej (CNN) '''
def define_model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

''' Ładowanie i przygotowanie danych CIFAR-10 '''
trainX, trainY, testX, testY = load_data()
trainX, testX = normalize_pixels(trainX, testX)

''' Definiowanie i kompilacja modelu '''
model = define_model()

''' Trenowanie modelu na danych treningowych '''
model.fit(trainX, trainY, epochs=50, batch_size=128, validation_split=0.2)

''' Ewaluacja modelu na danych testowych '''
_, acc = model.evaluate(testX, testY, verbose=0)

''' Wyświetlenie dokładności modelu '''
print('> %.3f' % (acc * 100.0))
