import numpy as np
import torch
import torch.nn as nn

import src.quantize as q


class NNUE(nn.Module):

    def __init__(self, input_size=768, hidden_size=256, output_size=1, max_value=1.0):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.max_value = max_value
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.clipped_relu(x)
        x = self.fc2(x)
        return x

    def clipped_relu(self, x):
        relu_output = nn.ReLU()(x)
        clipped_output = torch.clamp(relu_output, max=self.max_value)
        return clipped_output

    def squared_clipped_relu(self, x):
        relu_output = nn.ReLU()(x)
        clipped_output = torch.clamp(relu_output, max=self.max_value)
        squared_output = torch.pow(clipped_output, 2)
        return squared_output

    def quantize(self):
        self.fc1.weight = nn.Parameter(q.quantize_int16(self.fc1.weight), requires_grad=False)
        self.fc1.bias = nn.Parameter(q.quantize_int16(self.fc1.bias), requires_grad=False)
        self.fc2.weight = nn.Parameter(q.quantize_int16(self.fc2.weight), requires_grad=False)
        self.fc2.bias = nn.Parameter(q.quantize_int16(self.fc2.bias), requires_grad=False)
        self.max_value = q.MAX_INT16

    def dequantize(self):
        self.fc1.weight = nn.Parameter(q.dequantize_int16(self.fc1.weight), requires_grad=True)
        self.fc1.bias = nn.Parameter(q.dequantize_int16(self.fc1.bias), requires_grad=True)
        self.fc2.weight = nn.Parameter(q.dequantize_int16(self.fc2.weight), requires_grad=True)
        self.fc2.bias = nn.Parameter(q.dequantize_int16(self.fc2.bias), requires_grad=True)
        self.max_value = 1.0

    def save(self, file_path):
        self.quantize()
        with open(file_path, 'wb') as f:
            for param in [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]:
                param = param.cpu().detach().numpy().astype(np.int16)
                for value in param.flatten():
                    f.write(value.tobytes())
        self.dequantize()

    @staticmethod
    def load(file_path, input_size=768, hidden_size=256):
        nnue = NNUE(input_size=input_size, hidden_size=hidden_size)
        input_size = nnue.fc1.in_features
        hidden_size = nnue.fc1.out_features
        output_size = nnue.fc2.out_features

        with open(file_path, 'rb') as f:
            input_weight_size = input_size * hidden_size
            input_bias_size = hidden_size
            output_weight_size = hidden_size * output_size
            output_bias_size = output_size

            input_weights = np.frombuffer(f.read(input_weight_size * 2), dtype=np.int16).reshape(hidden_size, input_size)
            input_biases = np.frombuffer(f.read(input_bias_size * 2), dtype=np.int16)
            output_weights = np.frombuffer(f.read(output_weight_size * 2), dtype=np.int16).reshape(output_size, hidden_size)
            output_biases = np.frombuffer(f.read(output_bias_size * 2), dtype=np.int16)

            nnue.fc1.weight = nn.Parameter(torch.tensor(input_weights, dtype=torch.int16), requires_grad=False)
            nnue.fc1.bias = nn.Parameter(torch.tensor(input_biases, dtype=torch.int16), requires_grad=False)
            nnue.fc2.weight = nn.Parameter(torch.tensor(output_weights, dtype=torch.int16), requires_grad=False)
            nnue.fc2.bias = nn.Parameter(torch.tensor(output_biases, dtype=torch.int16), requires_grad=False)

        nnue.dequantize()
        return nnue

