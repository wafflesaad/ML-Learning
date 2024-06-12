import numpy as np
import matplotlib.pyplot as plt

def compute_model_output(w,x,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(0,m):
        f_wb[i] = w*x[i] + b

    return f_wb
# plt.style.use('./deeplearning.mplstyle')

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
