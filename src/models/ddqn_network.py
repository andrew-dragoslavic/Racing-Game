import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQNNetwork(nn.Module):
    def __init__(self, input_channels = 1, n_actions = 12, hidden_dim = 216):
        super(DDQNNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=7, stride=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(432,216)
        self.fc2 = nn.Linear(216, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
    
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)  # Keep batch dimension, flatten the rest
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output layer
        
        return x
    
    def save_checkpoint(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        state_dict = torch.load(filepath, weights_only=False)
        self.load_state_dict(state_dict)

    def get_action_values(self, state):
        with torch.no_grad():
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            return self.forward(state)
    
    def copy_weights_from(self, source_network):
        self.load_state_dict(source_network.state_dict())

    