import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras import models,layers

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.layers import Dense

X_train = np.array([[120, 2], [130, 3], [100, 2.2]])
y_train = np.array([[1],[0],[1]])

print(X_train.shape)
print(y_train.shape)

norm_1 = layers.Normalization(axis=-1)
norm_1.adapt(X_train)

Xn = norm_1(X_train)

Xt = np.tile(Xn,(100,1))
Yt = np.tile(y_train,(100,1))

print(Xt.shape)
print(Yt.shape)

model = models.Sequential(
    [
    keras.Input(shape=(2,)),
    layers.Dense(3, activation='sigmoid', name='Layer1') ,
    layers.Dense(1, activation='sigmoid', name='Layer2')]
)

w1,b1 =model.get_layer("Layer1").get_weights()
w2,b2 = model.get_layer("Layer2").get_weights()

print(f"{w1} . {b1} . {w2} . {b2}")

model.compile(
    loss = keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,            
    epochs=10,
)


w1,b1 =model.get_layer("Layer1").get_weights()
w2,b2 = model.get_layer("Layer2").get_weights()

print(f"{w1} . {b1} . {w2} . {b2}")

X_pred = np.array([[120, 2] , [400, 4]])
X_testn = norm_1(X_pred)
predictions = model.predict(X_testn)

pred = np.zeros(len(predictions))
print(len(pred))

for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        pred[i] = 1
    else:
        pred[i] = 0

print(pred)

print(predictions)