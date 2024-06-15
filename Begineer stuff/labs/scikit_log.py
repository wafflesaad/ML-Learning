import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X_train, y_train)

w =model.coef_
b = model.intercept_

print (f"{w}   {b}")

y_pred = model.predict(X_train)

print(y_pred)