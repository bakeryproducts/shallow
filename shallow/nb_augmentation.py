
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/augmentation.ipynb

import cv2
import torch
import albumentations as albu
import albumentations.pytorch as albu_pt

import cv2
import numpy as np
import matplotlib.pyplot as plt


class ToTensor(albu_pt.ToTensorV2):
    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask).permute((2,0,1))

    def apply(self, image, **params):
        return torch.from_numpy(image).permute((2,0,1))

def norm_aug(func):
    def norm(*args, **kwargs):
        mean=(0,0,0)
        #mean = (0.36718887, 0.3378791 , 0.31245533)
        std = (1,1,1)
        #std =(.5,.5,.5)
        #std = [4 * 0.09700591, 4 * 0.0953244 , 4 * 0.09326297]
        aug_func = func(*args, **kwargs)
        aug = albu.Compose([aug_func, albu.Normalize(mean=mean, std=std), ToTensor()])
        return aug
    return norm

def crop_aug(func):
    def crop(*args, **kwargs):
        aug_func = func(*args, **kwargs)
        size = kwargs['size']
        crop_aug = albu.OneOf([
                #albu.RandomResizedCrop(size, size, scale=(0.05, 0.4)),
                albu.RandomCrop(size,size)
                #albu.CropNonEmptyMaskIfExists(size, size)
            ], p=1)
        aug = albu.Compose([crop_aug, aug_func])
        return aug
    return crop

def to_gpu(t, device):
    return t.to(device)

@norm_aug
def get_val(*, size):
    return albu.Compose([albu.CenterCrop(size, size)])

@norm_aug
def get_val_forced(*, size):
    return albu.Compose([albu.CropNonEmptyMaskIfExists(size, size)])


@norm_aug
def get_test(*, size):
    return albu.Compose([albu.Resize(size, size)])

# def get_gpu_test(*, size):
#     return albu.Compose([albu.Resize(size, size)])

@crop_aug
@norm_aug
def get_light(*, size):
    return albu.Compose([albu.Flip(), albu.RandomRotate90()])


@crop_aug
@norm_aug
def get_medium(*, size):
    return albu.Compose([
                            albu.Flip(),
                            albu.ShiftScaleRotate(),  # border_mode=cv2.BORDER_CONSTANT
                            # Add occasion blur/sharpening
                            albu.OneOf([albu.GaussianBlur(), albu.IAASharpen(), albu.NoOp()]),
                            # Spatial-preserving augmentations:
                            # albu.OneOf([albu.CoarseDropout(), albu.MaskDropout(max_objects=5), albu.NoOp()]),
                            albu.GaussNoise(),
                            albu.OneOf(
                                [
                                    albu.RandomBrightnessContrast(),
                                    albu.CLAHE(),
                                    albu.HueSaturationValue(),
                                    albu.RGBShift(),
                                    albu.RandomGamma(),
                                ]
                            ),
                            # Weather effects
                            albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),
                        ])

@crop_aug
@norm_aug
def get_hard(*, size):
    return albu.Compose([
                            albu.RandomRotate90(),
                            albu.Transpose(),
                            albu.RandomGridShuffle(p=0.2),
                            albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.2),
                            albu.ElasticTransform(alpha_affine=5, p=0.2),
                            # Add occasion blur
                            albu.OneOf([albu.GaussianBlur(), albu.GaussNoise(), albu.IAAAdditiveGaussianNoise(), albu.NoOp()]),
                            # D4 Augmentations
                            albu.OneOf([albu.CoarseDropout(), albu.NoOp()]),
                            # Spatial-preserving augmentations:
                            albu.OneOf(
                                [
                                    albu.RandomBrightnessContrast(brightness_by_max=True),
                                    albu.CLAHE(),
                                    albu.HueSaturationValue(),
                                    albu.RGBShift(),
                                    albu.RandomGamma(),
                                    albu.NoOp(),
                                ]
                            ),
                            # Weather effects
                            albu.OneOf([albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1), albu.NoOp()]),
                        ])

def get_aug(aug_type="val", size=256):
    """aug_type (str): one of `val`, `test`, `light`, `medium`, `hard`
       size (int): final size of the crop"""

    types = {
        "val" : get_val,
        "test" : get_test,
        "gpu_test" : get_test,
        "light" : get_light,
        "medium" : get_medium,
        "hard": get_hard,
    }

    return types[aug_type](size=size)