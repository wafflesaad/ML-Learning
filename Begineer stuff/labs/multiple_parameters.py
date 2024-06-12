import numpy as np
import matplotlib.pyplot as plt
# size(sqft), No. of bedrooms, No. of floors, Age of house
x_train = np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
x_features = np.array(["size(sqft)", "No. of bedrooms", "No. of floors", "Age of house"])
#price
y_train = np.array([460, 232, 178])
print(x_train[:,1])
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()

def z_normalize(x):
    u = np.mean(x,axis=0)
    sd = np.std(x,axis=0)

    x_norm = (x - u)/sd

    return x_norm,u,sd

x_norm,u,sd = z_normalize(x_train)

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb = (np.dot(w,x[i]) + b)
        cost = cost + (f_wb - y[i])**2

    cost = cost / (2*m)

    return cost

def compute_gradient(x,y,w,b,m,n):
    dj_db = 0.0
    dj_dw = np.zeros(n)


    for i in range(m):
        err = (np.dot(w,x[i]) + b) - y[i]

        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i][j]

        dj_db = dj_db + err

    dj_db = dj_db / m
    dj_dw = dj_dw / m

    return dj_db, dj_dw

def gradient_descent(x,y,w_in,b_in,iter,a):
    m = x.shape[0]
    n = x[0].shape[0]
    j_history = []
    p_history = []
    w = w_in
    b = b_in

    for i in range(iter):
        dj_db, dj_dw = compute_gradient(x,y,w,b,m,n)
        b = b - a*dj_db
        w = w - a*dj_dw

        j_history.append(compute_cost(x,y,w,b))

        p_history.append([w,b])

    return w,b,j_history,p_history
        
def model_output(x,w,b):
    m = x.shape[0]
    
    y_pred = np.zeros(m)

    for i in range(m):
        y_pred[i] = np.dot(w,x[i]) + b
    

    return y_pred


alpha = 0.01
b_init = 10
w_init = np.array([1,1,1,1])


w,b,j_history, p_history =  gradient_descent(x_norm, y_train, w_init,b_init, 1000, alpha)
plt.plot(j_history)
plt.xlabel("iterations")
plt.ylabel("Cost")
plt.title("Learning curve")
plt.show()
print(f"w = {w}, b = {b}")

y_pred = model_output(x_norm,w,b)
print(y_pred)

to_pred = np.array([1000,2,2,50])

to_pred_norm = (to_pred - u)/sd

pred = np.dot(w, to_pred_norm) + b


print(pred)

