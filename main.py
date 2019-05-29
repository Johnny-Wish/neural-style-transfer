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
    projection_layer = VggProjection()
    style_tensor = ImageLoader(path=args.style_path).tensor
    content_image = ImageLoader(path=args.content_path)
    content_tensor = content_image.tensor
    styler = StyleNet(reference_model, projection_layer, content_tensor, style_tensor, [4], [1, 2, 3, 4])
    styler.build_model()
    session = Session(styler, Adam, alpha=args.alpha, n_steps=args.steps_per_epoch)

    for epoch_count in range(1, args.n_epochs + 1):
        print('starting epoch {}'.format(epoch_count))
        result = session.epoch()
        dump_path = os.path.join(args.output, 'pastiche_{}.jpg'.format(epoch_count))
        ImageDumper(result, dump_path).dump()
