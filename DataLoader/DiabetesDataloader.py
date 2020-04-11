from torch import nn
import torch
from torch import tensor
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
from torch.utils.data import Dataset, DataLoader
# Dataloader does the following
# Data->Shuffle->Cut it into batch sizes(Iterables)
# For i, data in enumerate(train_loader):
# inputs, labels = data #Unpack
# inputs, labels = Variable(inputs), Variable(labels) #Wrap
class DiabetesDataset(DataLoader):
    def __init__(self):
        dataset = np.loadtxt('diabetes.csv', delimiter=',', unpack=False,dtype=np.float32)
        self.x_data = torch.from_numpy(dataset[:,0:-1])
        self.y_data = torch.from_numpy(dataset[:,[-1]])
        self.length = dataset.shape[0]
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.length

dataset = DiabetesDataset()
train_loader = DataLoader(dataset = dataset, batch_size=32, shuffle=True)
#class definition
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

#Loss and criterion Initialisation
criterion = nn.BCELoss()
optimizer = torch.optim.Rprop(classifier.parameters(), lr=0.1)

#train loop
for epoch in range(10):
    for i,data in enumerate(train_loader, 0):
        x_val, y_val = data
        x_val, y_val = Variable(x_val), Variable(y_val)
        y_pred = classifier(x_val)
        l = criterion(y_pred, y_val)
        optimizer.zero_grad()
        print("Epoch: ", epoch, "Iteration: ", i, "Loss = ", l.data)
        l.backward()
        optimizer.step()