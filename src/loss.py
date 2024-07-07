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
        if lambda_ == 0.0:
            return torch.mean((inputs - targets) ** 2)
        elif lambda_ == 1.0:
            return torch.mean((inputs - targets) ** 2)
        else:
            result = targets[:, 0]
            score = targets[:, 1]
            loss_result = torch.mean((inputs - result) ** 2)
            loss_score = torch.mean((inputs - score) ** 2)
            loss_weighted = lambda_ * loss_score + (1 - lambda_) * loss_result
            return loss_weighted
