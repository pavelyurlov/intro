from torchvision import models


def get_lraspp_mobilenet_v3_large(num_classes):
    model = models.segmentation.lraspp_mobilenet_v3_large(num_classes=num_classes)

    model_dict = model.state_dict()

    # exclude weights for 21 classes
    pretrained_dict = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True).state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not ('low_classifier' in k or 'high_classifier' in k)}

    # upload remaining pretrained weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model
