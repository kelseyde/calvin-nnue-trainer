import torch.nn
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """
    A loss function that interpolates between the mean-squared-error loss of 1) the game result and 2) the score.

    Interpolation is governed by the lambda parameter:

    lambda_ = 0.0 - purely based on game results
    0.0 < lambda_ < 1.0 - interpolated score and result
    lambda_ = 1.0 - purely based on search scores
    """

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, inputs, targets, lambda_):
        result = targets[:, 0]
        score = targets[:, 1]
        loss_result = self.mean_squared_error(inputs, result)
        loss_score = self.mean_squared_error(inputs, score)
        loss_weighted = lambda_ * loss_score + (1 - lambda_) * loss_result
        return loss_weighted

    def mean_squared_error(self, inputs, targets):
        return torch.mean((inputs - targets) ** 2)

