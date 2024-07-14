import os

import numpy as np
import torch
import torch.nn as nn

import src.quantize as q


class NNUE(nn.Module):

    def __init__(self, input_size=768, hidden_size=256):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_data):
        stm = self.fc1(input_data[:, 0])                            # Accumulate the side-to-move features
        nstm = self.fc1(input_data[:, 1])                           # Accumulate the not-side-to-move features
        in_ = self.concat(stm, nstm)                                # Concatenate the stm/nstm features
        hidden = self.clipped_relu(in_)                             # Apply clipped ReLU activation function
        out_ = self.fc2(hidden)                                     # Pass hidden layer activations to output layer
        return out_

    def loss(self, prediction, targets, scale, lambda_):
        """
        Compute the loss function for the neural network. The loss function is the mean squared error between the
        expected wdl and the blend of the game result wdl and the centipawn score converted to wdl space.
        The blend is controlled by the lambda parameter: lambda=0.0 means fully based on game result, lambda=1.0 means
        fully based on cp score.
        """
        wdl_result, cp_eval = targets[:, 0], targets[:, 1]          # Extract game result (wdl) and score (cp)
        wdl_eval = torch.sigmoid(cp_eval / scale)                   # Convert score from cp to wdl
        expected = wdl_eval * (1 - lambda_) + wdl_result * lambda_  # Blend game result wdl and score wdl
        predicted = torch.sigmoid(prediction)                       # Convert nnue output to wdl
        loss = torch.mean((expected - predicted) ** 2)              # Compute MSE between expected and predicted wdl
        return loss

    def concat(self, stm, nstm):
        return torch.cat((stm, nstm), dim=1)

    def clipped_relu(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def save(self, file_path):
        input_weights = self.fc1.weight.cpu().detach().numpy()
        input_biases = self.fc1.bias.cpu().detach().numpy()
        output_weights = self.fc2.weight.cpu().detach().numpy()
        output_bias = self.fc2.bias.cpu().detach().numpy()

        # Quantize the parameters
        quant_w0, quant_b0, quant_w1, quant_b1 = q.quantize(input_weights, input_biases, output_weights, output_bias)

        # Write the quantized parameters to the file
        with open(file_path, 'wb') as f:
            for param in [quant_w0, quant_b0, quant_w1, quant_b1]:
                f.write(param.tobytes())


    @staticmethod
    def load(file_path, input_size=768, hidden_size=256):
        nnue = NNUE(input_size=input_size, hidden_size=hidden_size)

        file_size = os.path.getsize(file_path)
        bytes_read = 0

        with open(file_path, 'rb') as f:
            input_weight_size = input_size * hidden_size
            input_bias_size = hidden_size
            output_weight_size = (hidden_size * 2) * 1  # output_size is 1 in this case
            output_bias_size = 1

            w0 = np.frombuffer(f.read(input_weight_size * 2), dtype=np.int16).reshape(hidden_size, input_size)
            bytes_read += input_weight_size * 2
            b0 = np.frombuffer(f.read(input_bias_size * 2), dtype=np.int16)
            bytes_read += input_bias_size * 2
            w1 = np.frombuffer(f.read(output_weight_size * 2), dtype=np.int16).reshape(1, hidden_size * 2)
            bytes_read += output_weight_size * 2
            b1 = np.frombuffer(f.read(output_bias_size * 2), dtype=np.int16)
            bytes_read += output_bias_size * 2

            # if bytes_read != file_size:
            #     raise ValueError(f"Expected to read {file_size} bytes, but only read {bytes_read} bytes.")

            # Dequantize the weights and biases
            w0, b0, w1, b1 = q.dequantize(w0, b0, w1, b1)

            # Set the weights and biases to the model
            nnue.fc1.weight = nn.Parameter(torch.tensor(w0, dtype=torch.float32), requires_grad=True)
            nnue.fc1.bias = nn.Parameter(torch.tensor(b0, dtype=torch.float32), requires_grad=True)
            nnue.fc2.weight = nn.Parameter(torch.tensor(w1, dtype=torch.float32), requires_grad=True)
            nnue.fc2.bias = nn.Parameter(torch.tensor(b1, dtype=torch.float32), requires_grad=True)

        return nnue

