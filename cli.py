import torch
from argparse import ArgumentParser


class CustomizedParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        """
        customized parser for command line arguments
        :param args: see argparse.ArgumentParser.__init__()
        :param kwargs: see argparse.ArgumentParser.__init__()
        """
        super(CustomizedParser, self).__init__(*args, **kwargs)
        self.add_argument("--content", required=True, help="path to content image")
        self.add_argument("--style", required=True, help="path to style image")
        self.add_argument("--size", default=256, type=int, help="size of images, default=256")
        self.add_argument("--steps", default=50, type=int, help="steps per epoch, default=50")
        self.add_argument("--cuda", action="store_true", help="enable CUDA acceleration if possible")
        self.add_argument("--epochs", default=6, type=int, help="number of epochs in total, default=6")
        self.add_argument("--alpha", default=1e6, type=float, help="relative weight of style loss to content "
                                                                   "loss, default=1e6")
        self.add_argument("--scratch", action="store_true", help="if set, train the model from scratch instead of "
                                                                 "content image")
        self.add_argument("--output", default="./output", help="dir to store output images")
        self.add_argument("--preserve_size", action="store_true", help="if set, rescale pastiche to its original size "
                                                                       "when dumping")


class CustomizedArgs:
    def __init__(self, parser: CustomizedParser):
        """
        wrapper for parsed arguments, used for reliable IDE hints
        :param parser: parser of command line arguments
        """
        self._args = parser.parse_args()

    @property
    def content_path(self):
        return self._args.content

    @property
    def style_path(self):
        return self._args.style

    @property
    def image_size(self):
        return self._args.size

    @property
    def steps_per_epoch(self):
        return self._args.steps

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() and self._args.cuda else "cpu")

    @property
    def n_epochs(self):
        return self._args.epochs

    @property
    def alpha(self):
        return self._args.alpha

    @property
    def from_scratch(self):
        return self._args.scratch

    @property
    def output(self):
        return self._args.output

    @property
    def preserve_size(self):
        return self._args.preserve_size
