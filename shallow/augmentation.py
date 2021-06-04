from functools import partial

import torch  # to tensor transform
import albumentations as albu
import albumentations.pytorch as albu_pt

from shallow.augs_custom import *


BBOX_PARAMS = {
    "format":'coco',
    "label_fields":None,
    "min_area":0.0,
    "min_visibility":0.0,
}
def composer(using_boxes): return albu.Compose if not using_boxes else partial(albu.Compose, bbox_params=albu.BboxParams(**BBOX_PARAMS)) 

class ToTensor(albu_pt.ToTensorV2):
    def apply_to_mask(self, mask, **params): return torch.from_numpy(mask).permute((2,0,1))
    def apply(self, image, **params): return torch.from_numpy(image).permute((2,0,1))

def augmentations_zoo(key, augmentor, p=1):
    """
        this should be a class, but it requires p, so kinda ugly atm
    """
    if key == 'd4':          augs = augmentor.compose([albu.Flip(), albu.RandomRotate90()] )
    elif key == 'norm':      augs = augmentor.compose([albu.Normalize(mean=augmentor.mean, std=augmentor.std), ToTensor()] )
    elif key == 'rand_crop': augs = albu.RandomCrop(augmentor.crop_h, augmentor.crop_w) 
    elif key == 'resize':    augs = albu.Resize(augmentor.resize_h, augmentor.resize_w)
    elif key == 'scale':     augs = albu.ShiftScaleRotate(0.1, 0.2, 45, p=p)
    elif key == 'blur':      augs = albu.OneOf([albu.GaussianBlur()], p=p)
    elif key == 'gamma':     augs = AddGammaCorrection(p=p)
    elif key == 'cutout':    augs = albu.OneOf([
                                        albu.Cutout(**augmentor.cutout_params),
                                        albu.GridDropout(**augmentor.griddrop_params),
                                        albu.CoarseDropout(**augmentor.coarsedrop_params)
                                    ],p=p)
    elif key == 'multi_crop': augs = albu.OneOf([
                                        albu.CenterCrop(augmentor.crop_h, augmentor.crop_w, p=.2),
                                        albu.RandomCrop(augmentor.crop_h, augmentor.crop_w, p=.8),
                                    ], p=p)    
    elif key == 'color_jit': augs =  albu.OneOf([
                                        albu.HueSaturationValue(10,15,10),
                                        albu.CLAHE(clip_limit=4),
                                        albu.RandomBrightnessContrast(.4, .4),
                                        albu.ChannelShuffle(),
                                    ], p=p)
    return augs


class AugmentatorBase:
    def __init__(self, cfg, compose):
        self.cfg = cfg 
        self.resize_h, self.resize_w = self.cfg.RESIZE
        self.crop_h, self.crop_w = self.cfg.CROP
        self.crop_val_h, self.crop_val_w = self.cfg.CROP_VAL if self.cfg.CROP_VAL is not (0,) else seld.cfg.CROP
        self.compose = compose
        self.mean = self.cfg.MEAN 
        self.std = self.cfg.STD 
        self.az = partial(augmentations_zoo, augmentor=self)
    
    def get_aug(self, kind): return getattr(self, f'aug_{kind}', None)()
    def aug_val(self): return self.compose([albu.CenterCrop(self.crop_val_h, self.crop_val_w), self.az('norm')])
    def aug_test(self): return self.compose([self.az('resize'), self.az('norm')])



def get_aug(Aug_class, aug_type, transforms_cfg, using_boxes=False, tag=False):
    """ aug_type (str): one of `val`, `test`, `light`, `medium`, `hard`
        transforms_cfg (dict): part of main cfg
    """
    compose = albu.Compose #if not tag else partial(albu.Compose, additional_targets={'mask1':'mask', 'mask2':'mask'})
    auger = Aug_class(cfg=transforms_cfg, compose=compose)
    aug = auger.get_aug(aug_type)
    assert aug is not None, aug_type
    return  aug


