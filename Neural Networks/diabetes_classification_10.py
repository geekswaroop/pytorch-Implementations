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
        self.layer1 = nn.Linear(8,1024)
        self.layer2 = nn.Linear(1024,512)
        self.layer3 = nn.Linear(512,256)
        self.layer4 = nn.Linear(256,128)
        self.layer5 = nn.Linear(128,64)
        self.layer6 = nn.Linear(64,32)
        self.layer7 = nn.Linear(32,16)
        self.layer8 = nn.Linear(16,8)
        self.layer9 = nn.Linear(8, 4)
        self.layer10 = nn.Linear(4,1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out1 = self.act(self.layer1(x))
        out2 = self.act(self.layer2(out1))
        out3 = self.act(self.layer3(out2))
        out4 = self.act(self.layer4(out3))
        out5 = self.act(self.layer5(out4))
        out6 = self.act(self.layer6(out5))
        out7 = self.act(self.layer7(out6))
        out8 = self.act(self.layer8(out7))
        out9 = self.act(self.layer9(out8))
        out10 = self.act(self.layer10(out9))
        return out10

classifier = DiabetesClassifier()

#Criterion and Optimizer Setup
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)


#Training
#Forward->loss->(zero_grad_set)->backward->step
for epoch in range(500):
    y_pred = classifier(x_data)
    l = criterion(y_pred, y_data)
    optimizer.zero_grad()
    print("Epoch: ", epoch, "Loss = ", l.data)
    l.backward()
    optimizer.step()