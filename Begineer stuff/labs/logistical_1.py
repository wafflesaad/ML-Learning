import numpy as np
import matplotlib.pyplot as plt
import copy

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    n = x.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        z = np.dot(x[i], w) + b
        f_wb = sigmoid(z)
        diff = f_wb - y[i]
        # dj_dw += diff*x[i]
        for j in range(n):
            dj_dw[j] += diff*x[i,j]
        dj_db += diff

    dj_db = dj_db/m
    dj_dw = dj_dw/m
    return dj_dw, dj_db

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0.0
    
    for i in range(m):
        z = np.dot(x[i],w) + b
        f_wb = sigmoid(z)
        cost_i = -y[i]*np.log(f_wb) - (1 - y[i])*np.log(1 - f_wb) 
        cost += cost_i

    return cost / m



def gradient_descent(x,y,w_in,b_in,a,iter):
    w = copy.deepcopy(w_in)
    b = b_in
    j_history = []
    p_history = []

    for i in range(iter):
        dj_dw, dj_db = compute_gradient(x,y,w,b)
        w = w - a*dj_dw
        b = b - a*dj_db

        j_history.append(compute_cost(x,y,w,b))
        p_history.append([w,b])



    return w,b,j_history,p_history


alpha = 0.1
w_in = np.zeros(X_train.shape[1])
b_in = 0

w,b,j_history, p_history = gradient_descent(X_train,y_train,w_in,b_in,alpha, 10000)

plt.plot(j_history)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.show()

print(w,b)

