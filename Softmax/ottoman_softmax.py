from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
from torch import nn, optim, cuda

batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training Ottoman Model on {device}\n{"=" * 44}')

#Dataloader
class OttoDatasetTrain(Dataset):
    def __init__(self):
        xy = np.loadtxt('train.csv',delimiter=',',skiprows = 1, usecols = np.arange(1,94),dtype='float32')
        df = pd.read_csv('train.csv', sep = ',')
        df['target'] =  df['target'].map({'Class_1': 0, 'Class_2': 1,
                                          'Class_3': 2, 'Class_4': 4,
                                          'Class_5': 4, 'Class_6': 5,
                                          'Class_7': 6, 'Class_8': 7,
                                          'Class_9': 8})
        #df['target'] = df['target'].astype('float64')
        self.len = xy.shape[0]
        self.x_data= torch.from_numpy(xy[:,:])
        self.y_data = torch.tensor(df['target'].values)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
#Create dataloader
train_dataset = OttoDatasetTrain()
train_loader = DataLoader(dataset = train_dataset, batch_size=32, shuffle=True)


#Model Definition
class OttomanNet(nn.Module):
    def __init__(self):
        super(OttomanNet,self).__init__()
        self.l1 = nn.Linear(93, 46)
        self.l2 = nn.Linear(46,18)
        self.l3 = nn.Linear(18,9)
    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

model = OttomanNet()

#Criterion and Optimiser initialisation
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#Training
#Forward->Loss->(Zero Grad)->Backward->step
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        y_pred = model(data)
        loss = criterion(y_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Print Stats
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))

