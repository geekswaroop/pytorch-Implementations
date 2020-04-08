import numpy as np 
import matplotlib.pyplot as plt 

w = 1.0
x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]

w_list = []
mse_list = []
def forward(x):
    return w*x

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

for w in np.arange(0.0, 4.1, 0.1):
    print("Weight = ", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        l_sum+= loss(x_val, y_val)
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        print("\t", x_val, y_val, y_pred_val, l)
    print("Mean Square Error: ", l_sum/(len(x_data)))
    w_list.append(w)
    mse_list.append(l_sum/(len(x_data)))
    print("\n")

plt.plot(w_list, mse_list)
plt.xlabel("Weights")
plt.ylabel("Mean Squared Errors")
plt.show()