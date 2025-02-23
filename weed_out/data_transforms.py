import numbers
import random
import torchvision

import numpy as np
from PIL import Image, ImageOps
import torch



#Basic Resize Transformations (used until 23.04.)
class Resize_Image(object):
    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, gt_label, pseudo_labels):
        pseudo_labels_new=[]
        for entry in pseudo_labels:
            pseudo_labels_new.append(entry.resize(self.size, Image.NEAREST))
        return gt_label.resize(self.size, Image.NEAREST),pseudo_labels_new


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, gt_label, pseudo_labels=None):
        return torch.LongTensor(np.array(gt_label, dtype=np.int)),\
               [torch.LongTensor(np.array(entry, dtype=np.int)) for entry in pseudo_labels]


class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
