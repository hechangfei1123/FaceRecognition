import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable

def weights_init(m):
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias,0.1)

class PNet(nn.Module):
    def __init__(self):
        super(PNet,self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,10,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(10,16,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=1),
            nn.PReLU()
        )
        self.conv4_1 = nn.Conv2d(32,1,kernel_size=1,stride=1)
        self.conv4_2 = nn.Conv2d(32,4,kernel_size=1,stride=1)

        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        label = F.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        # offset = offset.reshape(-1,4)
        return label,offset

class RNet(nn.Module):
    def __init__(self,is_train=False,use_cuda=True):
        super(RNet,self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,28,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(28,48,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(48,64,kernel_size=2,stride=1),
            nn.PReLU()
        )
        self.conv4 = nn.Linear(64*2*2,128)
        self.prelu4 = nn.PReLU()
        self.conv5_1 = nn.Linear(128,1)
        self.conv5_2 = nn.Linear(128,4)

        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0),-1)
        x = self.conv4(x)
        x = self.prelu4(x)

        label = torch.sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)

        return label,offset

class ONet(nn.Module):
    def __init__(self,is_train=False,use_cuda=True):
        super(ONet,self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(32,64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,128,kernel_size=2,stride=1),
            nn.PReLU()
        )
        self.conv5 = nn.Linear(128*2*2,256)
        self.prelu5 = nn.PReLU()
        self.conv6_1 = nn.Linear(256,1)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0),-1)
        x = self.conv5(x)
        x = self.prelu5(x)

        det = torch.sigmoid(self.conv6_1(x))
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)

        # if self.is_train is True:
        return det,box,landmark