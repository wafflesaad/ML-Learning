import numpy as np
import matplotlib.pyplot as plt

def compute_model_output(w,x,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(0,m):
        f_wb[i] = w*x[i] + b

    return f_wb
# plt.style.use('./deeplearning.mplstyle')

def compute_cost(x,y,f_wb):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        cost_sum = cost_sum + (f_wb[i] - y[i])**2

    final_cost = cost_sum / (2*m)
    return final_cost

x_train = np.array([14.0,20.0,22.2, 15.5, 32.1])
y_train = np.array([1000,1300, 1200, 1150, 1500])
print(x_train)
print(y_train)

m = x_train.shape[0]
print(m)

plt.scatter(x_train,y_train, marker = "x", c="r")
plt.title("Housing prices")
plt.xlabel("Sq feet area");
plt.ylabel("Price")
plt.show()

w = 100
b = 100

temp_f_wb = compute_model_output(w,x_train,b )

plt.plot(x_train, temp_f_wb, marker="x", c="g")
plt.title("Housing prices prediction")
plt.xlabel("sq feet area")
plt.ylabel("price estimate")
plt.legend('LINE 1')
plt.show()

cost = compute_cost(x_train, y_train, temp_f_wb)
print(f"cost is {cost}")