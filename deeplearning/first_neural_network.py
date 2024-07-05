import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

X_train = np.array([[120, 2], [130, 3], [100, 2.2]])
y_train = np.array([[1],[0],[1]])


norm_1 = layers.Normalization(axis=-1)
norm_1.adapt(X_train)

Xn = norm_1(X_train)

def sigmoid(f):
    return 1 / (1 + np.exp(-f))

def dense(A_in, W, B):
    units = W.shape[1]
    A_out = np.zeros(units)

    for i in range(units):
        w = W[:,i]
        z = np.dot(A_in, w) + B[i]
        A_out[i] = sigmoid(z)

    return A_out

def sequential(A_in,W1, B1, W2, B2):  #network
    a1 = dense(A_in, W1, B1)
    a2 = dense(a1, W2, B2)
    return a2


W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

def predict(X, W1, B1, W2, B2):
    m = X.shape[0]
    p = np.zeros((m,1))

    for i in range(m):
        p[i,0] = sequential(X[i], W1, B1, W2, B2)

    return p

X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_1(X_tst)  # remember to normalize
predictions = predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

print(predictions)