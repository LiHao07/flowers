import sys
sys.path.append('/Users/lihao/miniconda3/lib/python3.6/site-packages')
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch


class FlowerModel(nn.Module):
    def __init__(self):
        super(FlowerModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        x = x = x.mean(3).mean(2)

        x = F.softmax(self.fc(x), dim=1)
        return x

if __name__ == '__main__':
    model = FlowerModel()
    x = torch.ones((1,3,128,128))
    pred = model(x)
    print(pred)

