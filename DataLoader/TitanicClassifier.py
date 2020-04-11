from torch import nn
import torch
from torch import tensor
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TitanicDataset(DataLoader):
    def __init__(self):
        xy = pd.read_csv('titanic.csv', delimiter=',')
        dropped_data =xy.drop(['Name','Embarked', 'Ticket', 'Embarked', 'Cabin'], axis=1 )
        dropped_data['Sex'] = dropped_data['Sex'].map({'female': 0, 'male': 0})
        prefinal_data=dropped_data.fillna(dropped_data.mean())
        final_data = prefinal_data.astype(np.float32)
        data = final_data.values    
        self.x_data = torch.from_numpy(data[:,2:])
        self.y_data = torch.from_numpy(data[:,[1]])
        self.length = data.shape[0]
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.length

dataset = TitanicDataset()
train_loader = DataLoader(dataset = dataset, batch_size=32, shuffle=True)

#model
class TitanicClassifier(nn.Module):
    def __init__(self):
        super(TitanicClassifier,self).__init__()
        self.layer1 = nn.Linear(6,60)
        self.layer2 = nn.Linear(60,10)
        self.layer3 = nn.Linear(10,1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out1 = self.act(self.layer1(x))
        out2 = self.act(self.layer2(out1))
        out3 = self.act(self.layer3(out2))
        return out3

classifier = TitanicClassifier()

criterion = nn.BCELoss()
optimizer = torch.optim.Rprop(classifier.parameters(), lr=0.1)

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