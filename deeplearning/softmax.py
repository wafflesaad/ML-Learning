import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

X_train = [[ 1.56 , 0.85],[-5.34 , 1.03],[-4.09 , 0.68],[-0.84 ,-1.95],[ 5.04 ,-2.92],[ 0.38 , 1.5 ]]
y_train = [2,0,0,1,3,2]

model = models.Sequential(
    [
        layers.Dense(units=25, activation="relu"),
        layers.Dense(units=15, activation="relu"),
        layers.Dense(units=4, activation="softmax")
    ]
)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(),
    optimizer= keras.optimizers.Adam(0.001)
)

model.fit(X_train,y_train, epochs=10)