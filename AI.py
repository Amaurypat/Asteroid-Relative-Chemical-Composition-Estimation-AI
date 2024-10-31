import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import scipy.integrate as integrate
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.regression import MeanAbsolutePercentageError
from scipy import interpolate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 620
hidden_size1 = 3100
hidden_size2 = 40000
hidden_size3 = 24800
hidden_size4 = 930
num_classes = 35
num_epochs = 100
batch_size = 10
learning_rate = 0.0001
losslist = []

cwd = os.getcwd()
os.chdir('samples/')
training = np.zeros(shape=(9000, 620))
traincomp = np.zeros(shape=(9000, 35))
samtest = np.zeros(shape=(1000, 620))
comptest = np.zeros(shape=(1000, 35))
for i in range(0,9000):
    training[i] = np.loadtxt(('sample%d' % i) + '.txt')
    traincomp[i] = np.loadtxt(('comp%d' % i) + '.txt')
for i in range(9001,6000):
    samtest[i-9001] = np.loadtxt(('sample%d' % i) + '.txt')
    comptest[i-9001] = np.loadtxt(('comp%d' % i) + '.txt')
training = (((torch.from_numpy(training)).to(torch.float32)).reshape(9000, 620)).to(device)
traincomp = (torch.from_numpy(traincomp).reshape(9000, 35)).to(device)

train_dataset = TensorDataset(training, traincomp)
test_dataset = TensorDataset((torch.from_numpy(samtest).to(torch.float32)).reshape(1000, 620), (torch.from_numpy(comptest).to(torch.float32)).reshape(1000, 35))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.l5 = nn.Linear(hidden_size4, num_classes)

    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        return out
    
try:
    os.chdir(cwd)
    model = NeuralNet()
    model.load_state_dict(torch.load('AI/AI.pth'))
    model.to(device)
    model.eval()
    print('Loading')
except:
    model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters())

criterion = nn.MSELoss()
os.chdir(cwd)
n_total_steps = 9000//batch_size  
for epoch in range(num_epochs):
    for i, (ttraining, ttraincomp) in enumerate(train_loader):
        ttraining = ttraining.to(device)
        ttraincomp = ttraincomp.to(device)

        outputs = model(ttraining)

        loss = (criterion(outputs.to(torch.float32), ttraincomp.to(torch.float32))).to(torch.float32)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    n_samples = len(test_loader)
    sum = 0
    with torch.no_grad():
        for i, (refl, comp) in enumerate(test_loader):
            refl = refl.to(device)
            comp = comp.to(device)
            outputs = model(refl)
            loss = (criterion(outputs, comp))*200
    losslist.append(float(eval(f'{loss.item():.4f}')))
    print(losslist)
print('Finished training')
torch.save(model.state_dict(), 'AI/AI.pth')
np.savetxt('Lossvalues.csv', losslist)

