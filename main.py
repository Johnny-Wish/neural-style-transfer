import os
from torch.optim import Adam
from torchvision.models.vgg import vgg16
from image_utils import ImageLoader, ImageDumper
from model import StyleNet
from session import Session
from projection import VggProjection
from cli import CustomizedArgs, CustomizedParser

if __name__ == '__main__':
    parser = CustomizedParser()
    args = CustomizedArgs(parser)

    reference_model = vgg16(pretrained=True).features
    projection_layer = VggProjection(device=args.device)
    style_tensor = ImageLoader(path=args.style_path, size=args.image_size, device=args.device).tensor
    content_image = ImageLoader(path=args.content_path, size=args.image_size, device=args.device)
    content_tensor = content_image.tensor
    styler = StyleNet(
        reference=reference_model,
        projection_layer=projection_layer,
        content=content_tensor,
        style=style_tensor,
        content_layer=[4],
        style_layer=[1, 2, 3, 4],
        device=args.device,
    )
    styler.build_model()
    print(styler.model)

    session = Session(
        styler=styler,
        optimizer_class=Adam,
        alpha=args.alpha,
        n_steps=args.steps_per_epoch,
        from_scratch=args.from_scratch,
    )

    for epoch_count in range(1, args.n_epochs + 1):
        print('starting epoch {}'.format(epoch_count))
        result = session.epoch()
        dump_path = os.path.join(args.output, 'pastiche_{}.jpg'.format(epoch_count))
        size = content_image.original_size if args.preserve_size else None
        ImageDumper(result, path=dump_path, size=size).dump()
