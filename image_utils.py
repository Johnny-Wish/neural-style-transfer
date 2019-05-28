import torch
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from PIL import Image


class ImageLoader:
    def __init__(self, path, size=(256, 256)):
        """
        tool for loading an image from disk and storing it in a tensor
        :param path: path to load the image
        :param size: size of image's width and height, used for resizing
        """
        img = Image.open(path)
        transformer = Compose([
            Resize(size),
            ToTensor(),
        ])
        # dummy batch size == 1
        self._tensor = transformer(img).unsqueeze(dim=0)  # type: torch.Tensor

    @property
    def tensor(self):
        return self._tensor


class ImageDumper:
    def __init__(self, tensor: torch.Tensor, path=None, size=None):
        """
        tool for dumping an image, before resizing it if necessary
        :param tensor: Tensor of shape (1, C, H, W)
        :param path: default path to dump the image
        :param size:
        """
        self.default_path = path
        if size is None:
            transformer = ToPILImage()
        else:
            transformer = Compose([
                torch.squeeze,
                ToPILImage(),
                Resize(size),
            ])

        self.image = transformer(tensor.to("cpu"))  # type: Image

    def dump(self, path=None):
        if path is None:
            path = self.default_path

        if path is None:
            print("image not saved, please specify a path")
        else:
            self.image.save(path)
