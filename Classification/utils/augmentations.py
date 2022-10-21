import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEANS = (0.485, 0.456, 0.406)
STDS = (0.229, 0.224, 0.225)


'''
def get_train_transforms_rand_aug(n_transformations=4,
                                  magnitude=5,
                                  sz=224,
                                  means=MEANS,
                                  stds=STDS):
    transforms = [A.HorizontalFlip(p=1),
                  A.Rotate(limit=30, p=1),
                  A.Perspective(p=1),
                  A.RandomBrightnessContrast(p=1),
                  A.CLAHE(p=1),
                  A.HueSaturationValue(p=1),
                  A.Blur(blur_limit=3, p=1)
                  ]
    composition = np.random.choice(transforms,
                                   size=n_transformations,
                                   replace=False)
    return A.Compose([A.Resize(height=sz, width=sz),
                      *composition,
                      A.Normalize(means, stds),
                      ToTensorV2()])
'''


def get_train_transforms_rand_aug(n_transformations=5,
                                  magnitude=5,
                                  resized_crop_size=224,
                                  means=MEANS,
                                  stds=STDS):
    transforms = [A.HorizontalFlip(p=1),
                  A.Rotate(limit=9 * magnitude, p=1),
                  A.Perspective(scale=(0.01 * magnitude, 0.02 * magnitude), p=1),
                  A.RandomBrightness(limit=0.03 * magnitude, p=1),
                  A.RandomContrast(limit=0.03 * magnitude, p=1),
                  A.Equalize(p=1),
                  A.Solarize(p=1),
                  A.Sharpen(p=1),
                  A.Posterize(p=1),
                  A.HueSaturationValue(hue_shift_limit=3 * magnitude,
                                       sat_shift_limit=3 * magnitude,
                                       val_shift_limit=3 * magnitude,
                                       p=1),
                  A.ChannelShuffle(p=1),
                  A.Blur(blur_limit=min(7, magnitude), p=1)
                  ]
    composition = np.random.choice(transforms,
                                   size=n_transformations,
                                   replace=False)
    return A.Compose([A.RandomResizedCrop(resized_crop_size, resized_crop_size),
                      *composition,
                      A.Normalize(means, stds),
                      ToTensorV2()])


def get_test_transforms(sz=224,
                        means=MEANS,
                        stds=STDS):
    test_transforms = A.Compose([
        A.Resize(height=sz, width=sz),
        A.Normalize(means, stds),
        ToTensorV2()
    ])
    return test_transforms


def get_train_transforms(sz=224,
                         means=MEANS,
                         stds=STDS):
    train_transforms = A.Compose([
        A.Resize(height=sz, width=sz),
        A.HorizontalFlip(),
        A.Perspective(),
        A.Rotate(limit=45),
        A.RandomBrightnessContrast(),
        A.CLAHE(),
        A.HueSaturationValue(),
        A.Blur(blur_limit=4),
        A.ChannelShuffle(p=0.2),
        A.Posterize(p=0.2),
        A.Normalize(means, stds),
        ToTensorV2()
    ])
    return train_transforms


def get_test_transforms_2(resized_size=256,
                          crop_size=224,
                          means=MEANS,
                          stds=STDS):
    test_transforms = A.Compose([
        A.Resize(height=resized_size, width=resized_size),
        A.CenterCrop(height=crop_size, width=crop_size),
        A.Normalize(means, stds),
        ToTensorV2()
    ])
    return test_transforms
