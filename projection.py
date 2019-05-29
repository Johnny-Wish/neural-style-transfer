import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    parent class for projecting input to the first layer of neural network
    """
    pass


DEFAULT_VGG_MEAN = (0.485, 0.456, 0.406)
DEFAULT_VGG_STD = (0.229, 0.224, 0.225)


class VggProjection(ProjectionLayer):
    def __init__(self, mean=DEFAULT_VGG_MEAN, std=DEFAULT_VGG_STD, device=None):
        super(VggProjection, self).__init__()
        # reshape mean and std to (C, H, W)
        self.mean = torch.Tensor(mean).view(-1, 1, 1).to(device=device)
        self.std = torch.Tensor(std).view(-1, 1, 1).to(device=device)

    def forward(self, input):
        # broadcasting mean and std to (N, C, H, W)
        return (input - self.mean) / self.std
