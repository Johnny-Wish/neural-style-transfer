# Data Science Course Project - Neural Style Transfer 
This is the final project of 2019 spring's Data Science course of Shuheng Liu (student ID 20154625) at Chongqing University. In this project is a simple PyTorch implementation of the neural style transfer technique proposed in [_Image Style Transfer Using Convolutional Neural Networks_](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

Experiment codes, environments, logs, metric curves, inputs, and ouputs can be found at this FloydHub [page](https://www.floydhub.com/wish1104/projects/style-transfer/jobs).

## Experimental Result

### Lake in Huxi campus of Chongqing University

Image generated in [this experiment](https://www.floydhub.com/wish1104/projects/style-transfer/35/), pastiche output can be found [here](https://www.floydhub.com/wish1104/projects/style-transfer/35/output/pastiche_100.jpg).

|                        Content Image                         |                         Style Image                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="images/lake.jpg" alt="content" width="256" height="256"/> | <img src="images/starry-night.jpg" alt="style" width="256" height="256"/> |
<img src="resources/pastiche-lake-10k-iters.jpg" alt="pastiche" width="512" height="512"/>

## Dependencies
- Python interpreter 3.6 or later
- torch (PyTorch) 1.0.1 or later
- torchvision 0.2.1 or later
- pillow 5.0.0 or later
__GPU device with CUDA acceleration is strongly recommended__

## Usage
```
 python main.py [-h] --content CONTENT --style STYLE [--size SIZE] 
                [--steps STEPS] [--cuda] [--epochs EPOCHS] [--alpha ALPHA] 
                [--scratch] [--output OUTPUT] [--preserve_size]
```

Example: 
```
python main.py --content /images/teaching-building.jpg --style /images/starry-night.jpg --size 512 --epochs 100 --steps 100 --alpha 5e7 --cuda --output /output
```

Usage of Arguments:
```
required arguments:
  --content CONTENT  path to content image 
  --style STYLE      path to style image

optional arguments:
  -h, --help         show this help message and exit
  --size SIZE        size of images, default=256, used for resizing content 
                     images, style image and the pastiche
  --steps STEPS      number of steps per epoch, default=50
  --cuda             enable CUDA acceleration if possible
  --epochs EPOCHS    number of epochs in total, default=6
  --alpha ALPHA      relative weight of style loss to content loss, default=1e6
  --scratch          if set, train the model from scratch instead of content 
                     image; could significatly slow down the training process, 
                     used with caution
  --output OUTPUT    dir to store output images; default='./output'; could 
                     result in ovewriting existent files
  --preserve_size    if set, rescale pastiche to its original size when dumping
```
