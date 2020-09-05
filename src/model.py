import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,num_classes):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,28,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(28,64,3)
        self.conv3 = nn.Conv2d(64,128,3)
        self.fc1   = nn.Linear(128*52*52, 256)
        self.fc2   = nn.Linear(256,num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
