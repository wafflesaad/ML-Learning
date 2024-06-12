import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

x_train = np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
x_features = np.array(["size(sqft)", "No. of bedrooms", "No. of floors", "Age of house"])
#price
y_train = np.array([460, 232, 178])

scaler = StandardScaler()
x_norm = scaler.fit_transform(x_train)

sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(x_norm,y_train)

w = sgdr.coef_
b = sgdr.intercept_

print(f"w = {w}, b = {b}")

print(sgdr)
