import random
from PIL import Image, ImageFilter
import numpy as np
import torch
import torchvision.transforms as T

def t_pretrain(cropSize):
    t = T.Compose([
        T.RandomResizedCrop(cropSize, scale=(0.2, 1.)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        #RandomFilter(),
        #GaussianNoise(0.25),
        #RandomMask(cropSize, 16, 0.7),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t

def t_finetune(cropSize):
    t = T.Compose([
        T.RandomResizedCrop(cropSize),
        # I saw one script use the transform below for SL. I think it is too simple, since it applies no scale or ratio
        # Test performance increases ~1% on IN100 when finetuning with below. I think it's info leak, since test images are cropped the same way
        #transforms.Resize(int(round(1.1428 * cropSize))),  # CIFAR: 1.1428 * 28 = 32, IN: 1.1428 * 224 = 256
        #transforms.RandomCrop(cropSize),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t

def t_test(cropSize):
    t = T.Compose([
        T.Resize(int(round(256 / 224 * cropSize))),  # CIFAR: 1.1428 * 28 = 32, IN: 1.1428 * 224 = 256
        T.CenterCrop(cropSize),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t

def t_tensor():
    t = T.ToTensor()
    return t

def t_tensor_aug(cropSize):
    t = T.Compose([
        T.RandomResizedCrop(cropSize, scale=(0.2, 1.), antialias=True),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        #T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5),
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class HighPassFilter(object):

    def __init__(self, sigma=[1., 5.]):
        self.lowPassFilter = GaussianBlur(sigma=sigma)

    def __call__(self, x):
        x = np.asarray(x).astype('float16') - np.asarray(self.lowPassFilter(x)).astype('float16') + 127
        x = Image.fromarray(np.clip(x, 0, 255).astype('uint8'))
        return x


class RandomFilter(object):

    def __init__(self, nFilters=1, maxMag=1, blurMag=0.06, kernelSize=9):
        # CIFAR: 1, 1, 0.3, 3 | ImageNet: 1, 1, 0.06, 9
        self.nFilters = nFilters
        self.maxMag = maxMag
        self.blurMag = blurMag
        self.kernelSize = kernelSize

    def __call__(self, x):
        for i in range(self.nFilters):
            kernel = np.random.uniform(low=0, high=self.blurMag, size=(1, 1, self.kernelSize, self.kernelSize))
            if self.maxMag is not None:
                kernel[0, 0, np.random.randint(self.kernelSize), np.random.randint(self.kernelSize)] = self.maxMag
            kernel = np.repeat(kernel, 3, axis=0)
            x = torch.nn.functional.conv2d(x, torch.from_numpy(kernel).float(), stride=1, groups=3, padding='same')
            x /= max(x.max(), 1e-6)
        return x


class GaussianNoise(object):

    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, x):
        x = x + torch.randn_like(x) * self.sigma ** 2
        x = torch.clamp(x, 0.0, 1.0)
        return x


class RandomMask(object):

    def __init__(self, cropSize, patchSize, maskProb):
        self.cropSize = cropSize
        self.patchSize = patchSize
        self.maskProb = maskProb

    def __call__(self, x):
        mask = np.random.choice([0, 1], replace=True, p=[self.maskProb, 1 - self.maskProb],
                                size=(1, int(self.cropSize / self.patchSize), int(self.cropSize / self.patchSize)))
        mask = np.repeat(np.repeat(mask, self.patchSize, axis=1), self.patchSize, axis=2)
        x = x * mask
        return x


class MultiCropTransform:
    """Take n random crops of one image as the query and key"""

    def __init__(self, nViews, baseTransform, nViews2=0, baseTransform2=None):
        self.nViews = nViews
        self.baseTransform = baseTransform
        self.nViews2 = nViews2
        self.baseTransform2 = baseTransform2

    def __call__(self, x):
        tList = [self.baseTransform(x) for _ in range(self.nViews)]
        if self.nViews2 > 0:
            tList = tList + [self.baseTransform2(x) for _ in range(self.nViews2)]

        return tList
