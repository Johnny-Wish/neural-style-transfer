import torch
import torch.nn as nn
import torch.nn.functional as F


class TransparentLossLayer(nn.Module):
    def __init__(self, target_features):
        """
        a transparent layer that, while allowing input to flow through unchanged, registers loss w.r.t. target
        :param target_features: used for computing loss
        """
        super(TransparentLossLayer, self).__init__()
        target = self._convert_features(target_features)
        # no gradient is computed on target
        self.target = target.detach()

    @staticmethod
    def _convert_features(features):
        raise NotImplementedError('_convert_features() not implemented')

    def forward(self, input_features):
        self.loss = F.mse_loss(input_features, self.target)
        return input_features


class ContentLoss(TransparentLossLayer):
    @staticmethod
    def _convert_features(features):
        return features


class StyleLoss(TransparentLossLayer):
    @staticmethod
    def _convert_features(features: torch.Tensor):
        batch_size, n_channels, height, width = features.shape
        n_rows = batch_size * n_channels
        n_cols = height * width
        matrix = features.view(n_rows, n_cols)
        # returns gram matrix normalized by element count
        return torch.mm(matrix, matrix.t()) / (n_rows * n_cols)
