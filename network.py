import torch
import torch.nn as nn
from clipped_relu import ClippedReLU

INPUT_LAYER_SIZE = 784
HIDDEN_LAYER_SIZE = 256

COLOUR_STRIDE = 64 * 6
PIECE_STRIDE = 64


class Network(nn.Module):

    def __init__(self, input_size=768, hidden_size=256, output_size=1, max_value=1.0):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.clipped_relu = ClippedReLU(max_value)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.clipped_relu(self.fc1(x))
        x = self.fc2(x)
        return x
