import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQNNetwork(nn.Module):
    """
    A Deep Double Q-Network (DDQN) implementation using PyTorch.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        pool1 (nn.MaxPool2d): First pooling layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool2 (nn.MaxPool2d): Second pooling layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output layer for action values.
    """

    def __init__(self, input_channels=1, n_actions=12, hidden_dim=216):
        """
        Initialize the DDQNNetwork.

        Args:
            input_channels (int): Number of input channels (default: 1).
            n_actions (int): Number of possible actions (default: 12).
            hidden_dim (int): Dimension of the hidden layer (default: 216).
        """
        super(DDQNNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=7, stride=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(432,216)
        self.fc2 = nn.Linear(216, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with action values for each input in the batch.
        """
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
        """
        Save the model's state dictionary to a file.

        Args:
            filepath (str): Path to save the checkpoint file.
        """
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        """
        Load the model's state dictionary from a file.

        Args:
            filepath (str): Path to the checkpoint file.
        """
        state_dict = torch.load(filepath, weights_only=False)
        self.load_state_dict(state_dict)

    def get_action_values(self, state):
        """
        Compute action values for a given state without updating the model.

        Args:
            state (torch.Tensor): Input state tensor. If 3D, it will be unsqueezed to 4D.

        Returns:
            torch.Tensor: Action values for the given state.
        """
        with torch.no_grad():
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            return self.forward(state)
    
    def copy_weights_from(self, source_network):
        """
        Copy weights from another DDQNNetwork instance.

        Args:
            source_network (DDQNNetwork): Source network to copy weights from.
        """
        self.load_state_dict(source_network.state_dict())

