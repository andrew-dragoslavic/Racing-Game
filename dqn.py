import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0) #32,20,20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0) #64,9,9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0) #64,7,7
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x