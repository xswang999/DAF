import random
import torchvision.transforms as transforms
from PIL import Image
import torch
from config.config import cfg


def get_transform(train=True):
    transform_list = []
    transform_list.append(transforms.Resize([224, 224]))

    if train:
        transform_list.append(transforms.RandomHorizontalFlip())

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize(mean=cfg.DATA_TRANSFORM.NORMALIZE_MEAN,
                                       std=cfg.DATA_TRANSFORM.NORMALIZE_STD)]


    transform_list += to_normalized_tensor

    return transforms.Compose(transform_list)