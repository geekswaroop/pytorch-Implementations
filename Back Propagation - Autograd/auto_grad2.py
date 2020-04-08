import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
b = 1
w1 = Variable(torch.Tensor([1.0]), requires_grad=True)
w2 = Variable(torch.Tensor([1.0]), requires_grad=True)

def forward(x):
    return w2*x*x + w1*x

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

print("Predicted value before training: ", forward(4))
for epoch in range(1000):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        grad1 = w1.grad.data[0]
        grad2 = w2.grad.data[0]
        print("Grad: " ,x_val, y_val, grad1, grad2)
        w1.data = w1.data - 0.01*grad1
        w2.data = w2.data - 0.01*grad2
        w1.grad.data.zero_()
        w2.grad.data.zero_()
    print("Epoch: ", epoch, "Loss: ", l)

print("Predicted value after training: ", forward(4).data[0])
print("W1: ", w1.data, "W2: ", w2.data)