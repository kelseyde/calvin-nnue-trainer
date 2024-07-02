import numpy as np
import torch
import torch.nn as nn


class NNUE(nn.Module):

    def __init__(self, input_size=768, hidden_size=256, output_size=1, max_value=1.0):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.max_value = max_value
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.squared_clipped_relu(x)
        x = self.fc2(x)
        return x

    def clipped_relu(self, x):
        return torch.clamp(torch.relu(x), max=self.max_value)

    def squared_clipped_relu(self, x):
        relu_output = self.relu(x)
        squared_output = torch.pow(relu_output, 2)
        clipped_output = torch.clamp(squared_output, max=self.max_value)
        return clipped_output

    def save(self, file_path):
        print(self.fc1.weight.detach())
        print(self.fc1.bias.detach())
        print(self.fc2.weight.detach())
        print(self.fc2.bias.detach())
        input_weights = self.fc1.weight.detach().numpy().astype(np.int16).flatten()
        input_biases = self.fc1.bias.detach().numpy().astype(np.int16).flatten()
        output_weights = self.fc2.weight.detach().numpy().astype(np.int16).flatten()
        output_bias = self.fc2.bias.detach().numpy().astype(np.int16).flatten()

        with open(file_path, 'wb') as f:
            f.write(input_weights.tobytes(order='C'))
            f.write(input_biases.tobytes(order='C'))
            f.write(output_weights.tobytes(order='C'))
            f.write(output_bias.tobytes(order='C'))
