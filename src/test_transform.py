import torchvision.transforms as transforms
import torch
from PIL import ImageFilter, Image

rand_size=224
rand_crop_scale=(0.3, 1.0)
color_jitter=1.0


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort

rand_transform = transforms.Compose([
    transforms.RandomResizedCrop(rand_size, scale=rand_crop_scale),
    transforms.RandomHorizontalFlip(),
    get_color_distortion(s=color_jitter),
    GaussianBlur(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225))
])

sample_img = Image.open('FID-300/ref/00001.png').convert('RGB')