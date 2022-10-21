import segmentation_models_pytorch as smp


def get_deeplab_v3_plus(num_classes):
    model = smp.DeepLabV3Plus(encoder_weights='imagenet', classes=num_classes)
    return model
