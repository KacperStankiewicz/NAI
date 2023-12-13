import numpy as np
import tensorflow as tf

'''
Tworzenie modelu sekwencyjnego przy użyciu Keras
=> Warstwa wejściowa (Input); 
=> Warstwa gęsta (Dense) z 10 neuronami i f-ją aktywacji ReLu;
=> Warstwa gęsta (Dense) z 8 neuronami i f-ją aktywacji sigmoid
'''
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(8, activation='sigmoid')
])

'''
Kompilacja modelu z optymalizatorem 'adam', funkcją straty 'binary_crossentropy'
i metryką 'accuracy'.
'''
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

''' Definiowanie danych treningowych x_train w formie tablicy '''
x_train = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
], dtype=np.float32)

''' Definiowanie etykiet (danych wyjściowych) y_train w formie tablicy '''
y_train = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
], dtype=np.float32)

''' Trenowanie modelu przy użyciu danych treningowych - 3000 epok. '''
model.fit(x_train, y_train, epochs=3000)

''' Ewaluacja modelu na podstawie danych treningowych '''
loss, accuracy = model.evaluate(x_train, y_train)

''' Wykonanie predykcji modelu na danych treningowych '''
predictions = model.predict(x_train)

''' 
Przekształcenie predykcji na postać binarną 
(1 jeśli wartość > 0.5, w przeciwnym razie 0)
'''
mapped_array = [[1 if elem > 0.5 else 0 for elem in inner_list] for inner_list in predictions]
print(mapped_array)
