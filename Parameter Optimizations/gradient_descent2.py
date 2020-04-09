# y_pred = x^2*w2 + x*w1 + b
import numpy as np 
import matplotlib.pyplot as plt 

w1 = 1.0
w2 = 1.0
b = 1.0 #No update
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1_list = []
w2_list = []
loss_list = []
def forward(x):
    return w2*x*x + x*w1

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

def gradient1(x,y):
    return 2*x*(x*x*w2 + x*w1  - y)

def gradient2(x,y):
    return 2*x*x*(x*x*w2 + x*w1 - y)

print("Prediction before training: ", forward(4))
for epoch in range(1000):
    for x_val, y_val in zip(x_data, y_data):
        grad1 = gradient1(x_val, y_val)
        grad2 = gradient2(x_val, y_val)
        print("\t", x_val, y_val, grad1, grad2)
        w1 = w1 - 0.01*grad1
        w2 = w2 - 0.01*grad2
        l = loss(x_val, y_val)
        # loss_list.append(l)
        # w_list.append(w)
    print("Progress: ", epoch, "W1 = ", w1, "W2 = ", w2, "Loss = ", l)
    w1_list.append(w1)
print("Predict after training: ", forward(4))
plt.plot(np.arange(1000), w1_list)
plt.xlabel("Epochs")
plt.ylabel("Weights")
plt.show()