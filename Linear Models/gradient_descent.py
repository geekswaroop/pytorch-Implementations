import numpy as np 
import matplotlib.pyplot as plt 

w = 1.0
x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]

w_list = []
loss_list = []
def forward(x):
    return w*x

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

def gradient(x,y):
    return 2*x*(x*w-y)

print("Prediction before training: ", forward(4))
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        print("\t", x_val, y_val, grad)
        w = w - 0.01*grad
        l = loss(x_val, y_val)
        # loss_list.append(l)
        # w_list.append(w)
    print("Progress: ", epoch, "W = ", w, "Loss = ", l)
    w_list.append(w)
print("Predict after training: ", forward(4))
plt.plot(np.arange(100), w_list)
plt.xlabel("Epochs")
plt.ylabel("Weights")
plt.show()