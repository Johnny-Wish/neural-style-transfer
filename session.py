import torch
from model import StyleNet


class Session:
    def __init__(self, styler: StyleNet, optimizer_class, alpha=1e6, n_steps=300, from_scratch=False):
        """
        a session than returns pastiche each epoch (and each step)
        :param styler: a StyleNet with specified content and style
        :param optimizer_class: a subclass of torch.optim.optimizer.Optimizer, whose constructor will be used
        :param alpha: relative weight of style loss to content loss
        :param n_steps: number of steps per epoch
        :param from_scratch: if True, pastiche is initialized to random noise; a clone of content image otherwise
        """
        self.styler = styler
        if from_scratch:
            device = self.styler.content_tensor.device
            self._pastiche = torch.rand(*self.styler.style_tensor.shape, device=device).requires_grad_(True)
        else:
            self._pastiche = self.styler.content_tensor.clone().requires_grad_(True)
        self.optimizer = optimizer_class([self._pastiche])
        self.alpha = alpha
        self._global_step = 0
        self.steps_per_epoch = n_steps

    def epoch(self):
        for _ in range(self.steps_per_epoch):
            self.step()
        return self._pastiche

    def step(self):
        self._global_step += 1
        self.optimizer.zero_grad()
        # feed the model forward
        self.styler.model(self._pastiche)
        # fetch and weight both losses
        style_loss = self.styler.style_loss
        content_loss = self.styler.content_loss
        print('style_loss = {}, content_loss = {}'.format(style_loss, content_loss))
        total_loss = content_loss + style_loss * self.alpha
        # backprop and do an optimization step
        total_loss.backward()
        self.optimizer.step(closure=None)
        # clamp pixel value to [0, 1]
        self._pastiche.data.clamp_(0, 1)
        return self._pastiche

    @property
    def pastiche(self):
        return self._pastiche
