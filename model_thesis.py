#%%
import h5py
f = h5py.File('/Users/stingcui/PycharmProjects/Deep Learning/skyimager.hdf5', 'r')
print(list(f.keys()))
print(f['dataset_1'].shape)
print(f['dataset_1_pv_instant_output'].shape)
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
# loader_params = {'batch_size': 100, 'shuffle': False, 'num_workers': 6}
# data_loader = data.DataLoader(dataset, **loader_params)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
#num_epochs = 100
#batch_size = 200
#learning_rate = 0.001

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 12, 3, 1)
        self.pool = nn.MaxPool2d(4, 4) # aufpassen
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(12*3*3, 84)
        self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        ## 1st convolution block
        x = self.conv1(x)
        x = F.relu(x)
        # x = nn.BatchNorm2d()
        x = self.pool(x)

        ## 2nd convolution block
        x = self.conv2(x)
        x = F.relu(x)
        # x = nn.BatchNorm2d()
        x = self.pool(x)

        ## two fully connected nets
        x = self.dropout1(x) # do we need this?
        x = self.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x) # do we need this? # im model.eval() kein dropout drin nur im model.train()
        x = self.fc2(x)
        return x
#%%
import numpy as np
Model = ConvNet()
a = np.array(f['dataset_1'])
b = np.array(f['dataset_1_pv_instant_output'])
dataset_1 = torch.from_numpy(a) # convert data into pytorch tensors
dataset_2 = torch.from_numpy(b) # convert data into pytorch tensors
#%%
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
class PV_Dataset(Dataset):
    def __init__(self):
        self.x = dataset_1
        self.y = dataset_2
        self.n_samples = dataset_1.shape[0] # number of samples is 19386
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

pv_dataset = PV_Dataset()
#%%
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# training pipeline
# 1: design model (input size, output size, forward pass)
# 2: construct loss and optimizer
# 3: training loop: 1. forward pass: compute prediction 2. backward pass: gradients 3. update weights

train_set = f['dataset_1'][0:900]
train_set = np.array(train_set)
test_set = f['dataset_1'][900:1000]
test_set = np.array(test_set)

train_set_output = f['dataset_1_pv_instant_output'][0:900]
train_set_output = np.array(train_set_output)
test_set_output = f['dataset_1_pv_instant_output'][900:1000]
test_set_set_output = np.array(test_set_output)

train_set = torch.from_numpy(train_set)
train_set = torch.permute(train_set, (0, 3, 1, 2))
test_set = torch.from_numpy(test_set)
test_set = torch.permute(test_set, (0, 3, 1, 2))
train_set_output = torch.from_numpy(train_set_output)
test_set_set_output = torch.from_numpy(test_set_set_output)
#%%
class PV_Dataset_example(Dataset):
    def __init__(self):
        self.x = train_set
        self.y = train_set_output
        self.n_samples = train_set.shape[0] # number of samples is 19386
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class PV_Dataset_test(Dataset):
    def __init__(self):
        self.x = test_set
        self.y = test_set_output
        self.n_samples = test_set.shape[0] # number of samples is 19386
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
#%%
pv_dataset_example = PV_Dataset_example()
pv_dataset_example_test = PV_Dataset_test()
pv_example_trainloader = DataLoader(dataset=pv_dataset_example, batch_size=4, shuffle=True)
pv_example_testloader = DataLoader(dataset=pv_dataset_example_test, batch_size=4, shuffle=True)
#%%
def train(model, device, data_loader, optimizer, epoch):
    model.train()
    for i, (data, target) in enumerate(data_loader):
        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
        print(data, target)
        optimizer.zero_grad()
        #import pdb
        #pdb.set_trace()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(data_loader.dataset),
                100. * i / len(data_loader), loss.item()))

def test(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device,dtype=torch.float32), target.to(device,dtype=torch.float32)
            output = model(data)
            # sum up batch loss
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset) #compute average test loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(data_loader.dataset),
    100. * correct / len(data_loader.dataset)))
#%%
# set options and train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_skyimager = ConvNet()
model_skyimager.to(device, dtype=torch.float32)
epoch = 2
learning_rate = 0.001
#%%
# testloader is lacking, the rest looks good, see set options and train
train_loader = pv_example_trainloader
test_loader = pv_example_testloader
optimizer = optim.Adam(model_skyimager.parameters(), lr=learning_rate) #momentum=momentum, what is momentum?
#%%
for epoch in range(1, epoch+1):
    train(model_skyimager, device, train_loader, optimizer, epoch)
    test(model_skyimager, device, test_loader)

# if (args.save_model):
# torch.save(model.state_dict(),"/results/mnist_cnn.pt")