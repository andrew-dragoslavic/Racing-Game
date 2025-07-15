import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(4, 6, kernel_size=7, stride=3, padding=0) #32,20,20
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4, stride=1, padding=0) #64,9,9
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(12*4*6, 216)
        self.fc2 = nn.Linear(216, num_actions)

    def forward(self, x):
        x = x[:, :, :81, :]

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x