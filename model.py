import torch
import torch.nn as nn
from projection import ProjectionLayer
from loss import ContentLoss, StyleLoss, TransparentLossLayer


class StyleNet:
    def __init__(self, reference, projection_layer, content, style, content_layer, style_layer, device=None):
        """
        a style network with properties `model`, `content_loss`, and `style_loss`
        :param reference: nn.Module, a reference CNN model used for feature extraction
        :param projection_layer: nn.Module, projection layer used for preprocessing input tensor
        :param content: torch.Tensor, tensor for content image
        :param style: torch.Tensor, tensor for style image
        :param content_layer: list[int], indices (starting from 1) where transparent ContentLoss will be inserted
        :param style_layer: list[int], indices (starting from 1) where transparent StyleLoss will be inserted
        :param device: torch.device, used for casting model and tensors
        """
        self.device = device
        self.reference = reference.to(device=self.device)  # type: nn.Module
        self.projection = projection_layer  # type: ProjectionLayer
        self.content_tensor = content  # type: torch.Tensor
        self.style_tensor = style  # type: torch.Tensor
        self.content_layer = content_layer  # type: list[int]
        self.style_layer = style_layer  # type: list[int]
        self._model: nn.Module = None
        self.layer_count: int = 0
        self._content_losses: list[torch.Tensor] = []
        self._style_losses: list[torch.Tensor] = []

    @property
    def model(self):
        if self._model is None:
            self.build_model()
        return self._model

    @property
    def content_loss(self):
        if self._model is None:
            self.build_model()
        return torch.stack([l.loss for l in self._content_losses]).sum()

    @property
    def style_loss(self):
        if self._model is None:
            self.build_model()
        return torch.stack([l.loss for l in self._style_losses]).sum()

    def _initialize_model(self):
        self._model = nn.Sequential().to(device=self.device)
        self._model.add_module('projection', self.projection)
        self.layer_count = 0
        self._content_losses = []
        self._style_losses = []

    def build_model(self):
        self._initialize_model()
        for layer in self.reference.children():
            self._adapt_named_layer(layer)
            self._attempt_insert_content_loss()
            self._attempt_insert_style_loss()
        # trim trailing layers after the last TransparentLossLayer
        self._trim_extra_layers()

    def _adapt_named_layer(self, layer):
        name = layer.__class__.__name__.lower()
        if name.startswith('conv'):
            self.layer_count += 1
        elif name.startswith('relu'):
            layer = nn.ReLU(inplace=False)
        self._model.add_module("{}_{}".format(name, self.layer_count), layer)

    def _attempt_insert_content_loss(self):
        if self.layer_count in self.content_layer:
            self.content_layer.remove(self.layer_count)
            target_features = self._model(self.content_tensor).detach()
            layer = ContentLoss(target_features)
            self._model.add_module('content_loss_{}'.format(self.layer_count), layer)
            self._content_losses.append(layer)

    def _attempt_insert_style_loss(self):
        if self.layer_count in self.style_layer:
            self.style_layer.remove(self.layer_count)
            target_features = self._model(self.style_tensor).detach()
            layer = StyleLoss(target_features)
            self._model.add_module('style_loss_{}'.format(self.layer_count), layer)
            self._style_losses.append(layer)

    def _trim_extra_layers(self):
        for last in range(len(self._model) - 1, -1, -1):
            if isinstance(self._model[last], TransparentLossLayer):
                break

        self._model = self._model[: last + 1]
