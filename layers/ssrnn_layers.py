import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Layers for input and hidden transformations
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden):
        # Compute transformations for input and hidden state
        input_transformation = self.input_layer(x)
        hidden_transformation = self.hidden_layer(hidden)

        # Apply non-linearity (ReLU is used as in the original implementation)
        activation = torch.relu(input_transformation + hidden_transformation)

        # Apply ReLU to get the output
        out = torch.relu(activation + hidden_transformation)
        return out