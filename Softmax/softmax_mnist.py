from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# MNIST Data Loading

train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)
    def forward(self, x):
        x = x.view(-1, 784) #Flatten the data
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        y_pred = F.relu(self.l5(x))
        return y_pred;

model = MNISTClassifier()
#GPU/CPU setup
model.to(device)

#Criterion and Optimizer definition
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#Training Function
#Forward->Loss->(set zero grad)->Backward->step
def train(epochs):
    model.train()
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


#Testing Function!

def test():
    model.eval()
    test_loss=0 #Running variable
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss+=criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1] #Outputs the value with the highest Log Probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss_avg = test_loss/(len(test_loader.dataset))
    accuracy = 100*(correct/len(test_loader.dataset))
    print("=====================")
    print("Average Loss: ", loss_avg, "Accuracy: ", accuracy)

#Main function (Boiler Plate, Can be used for all scripts!)
if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 5):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
