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
        self.add_argument("--size", default=256, help="size of images, default=256")
        self.add_argument("--steps", default=50, type=int, help="steps per epoch, default=50")
        self.add_argument("--cuda", action="store_true", help="enable CUDA acceleration if possible")
        self.add_argument("--epochs", default=6, type=int, help="number of epochs in total, default=6")
        self.add_argument("--alpha", default=1e6, type=float, help="relative weight of style loss to content "
                                                                   "loss, default=1e6")


class CustomizedArgs:
    def __init__(self, parser: CustomizedParser):
        """
        wrapper for parsed arguments, used for reliable IDE hints
        :param parser: parser of command line arguments
        """
        self.args = parser.parse_args()

    @property
    def content_path(self):
        return self.args.content

    @property
    def style_path(self):
        return self.args.style

    @property
    def image_size(self):
        return self.args.size

    @property
    def steps_per_epoch(self):
        return self.args.steps

    @property
    def cuda(self):
        return self.args.cuda

    @property
    def n_epochs(self):
        return self.args.epochs

    @property
    def alpha(self):
        return self.args.alpha
