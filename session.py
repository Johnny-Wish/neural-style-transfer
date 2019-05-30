import torch
from model import StyleNet
from global_utils import flush_json_metrics


class Session:
    def __init__(self, styler: StyleNet, optimizer_class, alpha=1e6, n_steps=300, start_from="content"):
        """
        a session than returns pastiche each epoch (and each step)
        :param styler: a StyleNet with specified content and style
        :param optimizer_class: a subclass of torch.optim.optimizer.Optimizer, whose constructor will be used
        :param alpha: relative weight of style loss to content loss
        :param n_steps: number of steps per epoch
        :param start_from: if 'content', start from a clone of content
                           if 'style', start from a clone of style
                           if 'scratch', start from a clone of style
                           if a torch.Tensor, start from the tensor
        """
        self.styler = styler
        self.metrics = {}
        if start_from == "scratch":
            device = self.styler.content_tensor.device
            self._pastiche = torch.rand(*self.styler.style_tensor.shape, device=device).requires_grad_(True)
        elif start_from == "content":
            self._pastiche = self.styler.content_tensor.clone().requires_grad_(True)
        elif start_from == "style":
            self._pastiche = self.styler.style_tensor.clone().requires_grad_(True)
        else:
            self._pastiche = start_from.clone().requires_grad_(True)
        self.optimizer = optimizer_class([self._pastiche])
        self.alpha = alpha
        self._global_step = 0
        self.steps_per_epoch = n_steps

    def epoch(self):
        for _ in range(self.steps_per_epoch):
            self.step()
        flush_json_metrics(self.metrics, step=self._global_step)
        return self._pastiche

    def step(self):
        self._global_step += 1
        self.optimizer.zero_grad()
        # feed the model forward
        self.styler.model(self._pastiche)
        # fetch and weight both losses
        style_loss = self.styler.style_loss
        content_loss = self.styler.content_loss
        total_loss = content_loss + style_loss * self.alpha
        self.metrics = {'style_loss': style_loss, 'content_loss': content_loss, 'total_weighted_loss': total_loss}
        # backprop and do an optimization step
        total_loss.backward()
        self.optimizer.step(closure=None)
        # clamp pixel value to [0, 1]
        self._pastiche.data.clamp_(0, 1)
        return self._pastiche

    @property
    def pastiche(self):
        return self._pastiche
