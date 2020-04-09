from torch import nn
import torch
from torch import tensor
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
x_data = Variable(tensor([[1.0], [2.0], [3.0],[4.0]]))
y_data = Variable(tensor([[0.0], [0.0], [1.0],[1.0]]))

loss_list = []
# Model Class definition
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = Model()

#Criterion and Optimizer Initialisation
criterion = nn.BCELoss(size_average=None)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#Training
#Forward->loss->(Zero Grad)->backward->update
for epoch in range(500):
    y_pred = model(x_data)
    print("x_data= ", x_data, "y_pred= ", y_pred)

    l = criterion(y_pred, y_data)
    print("Epoch= ", epoch, "Loss= ", l.data)
    loss_list.append(l.data)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
# print(model(1.0).data.item()>0.5)
# print(model(7.0).data.item()>0.5)
plt.plot(np.arange(500), loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()    