from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" * 44}')

#Dataloader
class OttoDatasetTrain(Dataset):
    def __init__(self):
        xy = np.loadtxt('train.csv',delimiter=',',skiprows = 1, usecols = np.arange(1,94))
        df = pd.read_csv('train.csv', sep = ',')
        df['target'] =  df['target'].map({'Class_1': 1, 'Class_2': 2,
                                          'Class_3': 3, 'Class_4': 4,
                                          'Class_5': 5, 'Class_6': 6,
                                          'Class_7': 7, 'Class_8': 8,
                                          'Class_9': 9})
        df['target'] = df['target'].astype('float64')
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:])
        self.y_data = torch.tensor(df['target'].values)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

#Custom Dataloader for test dataset
class OttoDatasetTest(Dataset):
    def __init__(self):
        xy = np.loadtxt('test.csv',delimiter=',',skiprows = 1, usecols = np.arange(1,94))
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:])
        #self.y_data = torch.tensor(df['target'].values)

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len
