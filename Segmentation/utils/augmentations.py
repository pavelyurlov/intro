import albumentations as A
from albumentations.pytorch import ToTensorV2


MEANS = (0.485, 0.456, 0.406)
STDS = (0.229, 0.224, 0.225)


def get_train_transforms(resized_height=528,
                         resized_width=928,
                         means=MEANS,
                         stds=STDS):
    train_transforms = A.Compose([
        A.Resize(height=resized_height, width=resized_width),
        A.HorizontalFlip(),
        A.VerticalFlip(),
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


def get_test_transforms(resized_height=528,
                        resized_width=928,
                        means=MEANS,
                        stds=STDS):
    test_transforms = A.Compose([
        A.Resize(height=resized_height, width=resized_width),
        A.Normalize(means, stds),
        ToTensorV2()
    ])
    return test_transforms


def transform_to_tensor():
    return A.Compose([ToTensorV2()])
