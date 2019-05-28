import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    parent class for projecting input to the first layer of neural network
    """
    pass


class VggProjection(ProjectionLayer):
    def __init__(self, mean, std):
        super(VggProjection, self).__init__()
        # reshape mean and std to (C, H, W)
        self.mean = torch.Tensor(mean).view(-1, 1, 1)
        self.std = torch.Tensor(std).view(-1, 1, 1)

    def forward(self, input):
        # broadcasting mean and std to (N, C, H, W)
        return (input - self.mean) / self.std
