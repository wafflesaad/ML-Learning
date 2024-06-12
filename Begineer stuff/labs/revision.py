import numpy as np
import matplotlib.pyplot as plt


x_train = np.array([1.0,3.0, 4, 6])
y_train = np.array([2.0,4.0, 4.2, 4.9])


def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w*x[i] + b
        cost_sum = cost_sum + (f_wb - y)**2

    return cost_sum


def compute_gradient(x,y,w,b,m):
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        dj_dw = dj_dw + ((w*x[i] + b) - y[i])*x[i]
        dj_db = dj_db + ((w*x[i] + b) - y[i])

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, a, w, b, iter):
    m = x.shape[0]

    j_history = []
    p_history = []

    for i in range(iter):
        dj_dw, dj_db = compute_gradient(x,y,w,b,m)
        tmp_w = w - a * dj_dw
        tmp_b = b - a * dj_db
        w = tmp_w
        b = tmp_b

        if i < 1000:
            j_history.append(compute_cost(x,y,w,b))
            p_history.append([w,b])
        


    return w, b, j_history , p_history

def model_outputs(x,w,b):
    m = x.shape[0]
    y_pred = np.zeros(m)

    for i in range(m):
        y_pred[i] = w*x[i] + b

    return y_pred

alpha = 0.01

w_in=100
b_in=100

w, b, j_history, p_history = gradient_descent(x_train, y_train, alpha, w_in, b_in, 100000)

print("W= ", w , " B= " , b)

x = np.array([7.0, 8.0, 10.0, 12.0, 14.0])

y = model_outputs(x_train, w, b)

print(y)

plt.scatter(x_train, y_train, marker="X")
plt.xlabel("Sq feet")
plt.ylabel("Price")
plt.title("Housing prices")

plt.plot(x_train, y, color="red")

plt.show()

