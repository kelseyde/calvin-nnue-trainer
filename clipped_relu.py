import torch
import torch
import torch.nn as nn


class ClippedReLU(nn.Module):

    def __init__(self, max_value=1.0):
        super(ClippedReLU, self).__init__()
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(torch.relu(x), max=self.max_value)