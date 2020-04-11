from torch import nn
import torch
from torch import tensor
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np

#load Dataset
data = np.loadtxt('diabetes.csv', delimiter=',', unpack=False,dtype=np.float32)
x_data = Variable(torch.from_numpy(data[:, 0:-1]))
y_data = Variable(torch.from_numpy(data[:, [-1]]))

print(y_data.shape)
print(x_data.shape)

#Model Definition
class DiabetesClassifier(nn.Module):
    def __init__(self):
        super(DiabetesClassifier,self).__init__()
        self.layer1 = nn.Linear(8,60)
        self.layer2 = nn.Linear(60,10)
        self.layer3 = nn.Linear(10,1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out1 = self.act(self.layer1(x))
        out2 = self.act(self.layer2(out1))
        out3 = self.act(self.layer3(out2))
        return out3

classifier = DiabetesClassifier()

#Criterion and Optimizer Setup
criterion = nn.BCELoss()
optimizer = torch.optim.Rprop(classifier.parameters(), lr=0.1)


#Training
#Forward->loss->(zero_grad_set)->backward->step
for epoch in range(500):
    y_pred = classifier(x_data)
    l = criterion(y_pred, y_data)
    optimizer.zero_grad()
    print("Epoch: ", epoch, "Loss = ", l.data)
    l.backward()
    optimizer.step()


