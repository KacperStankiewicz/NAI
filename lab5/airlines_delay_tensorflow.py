"""
Autorzy
Rutkowski Marcin - s12497
Stankiewicz Kacper - s22619

Predykcja opóźnienia lotu
"""

import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''Wczytywanie danych z pliku csv'''
data = pd.read_csv('data/airlines_delay_train.csv')

'''Podział danych na cechy (X) i etykiety (Y)'''
X = data[['Flight', 'Time', 'Length', 'DayOfWeek']]
Y = data['Class']

'''Podział danyhc na zestawy treningowe i testowe'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

'''Standaryzacja danych'''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''Definicja modelu TensorFlow'''
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

'''Kompilacja modelu'''
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

'''Trenowanie modelu'''
model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split=0.2)

'''Ocena modelu na zestawie testowym'''
_, acc = model.evaluate(X_test, Y_test)
print(f'Accuracy on test data: {acc}')

'''Przewidywanie na podstawie modelu'''
predictions = model.predict(X_test)

rounded_predictions = tf.round(predictions)

cm = confusion_matrix(Y_test, rounded_predictions)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()
