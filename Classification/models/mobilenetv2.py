import torch
import torch.nn as nn


def _make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def conv_norm_activation(input_dim, output_dim, kernel_size, stride=1, groups=1):
    layers = [
        nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, groups=groups, bias=False),
        nn.BatchNorm2d(output_dim),
        nn.ReLU6(inplace=True)
    ]
    return layers


class InvertedResidual(nn.Module):
    def __init__(self, input_dim, output_dim, stride, expansion):
        super().__init__()

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(input_dim * expansion))
        self.use_res_connect = stride == 1 and input_dim == output_dim

        layers = []
        if expansion != 1:
            layers.extend(
                *conv_norm_activation(input_dim, hidden_dim, 1)
            )
        layers.extend(
            [
                *conv_norm_activation(hidden_dim, hidden_dim, 3, stride, hidden_dim),
                nn.Conv2d(hidden_dim, output_dim, 1, bias=False),
                nn.BatchNorm2d(output_dim)
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MyMobileNetV2(nn.Module):
    def __init__(self, num_classes, inverted_residual_setting=None,
                 width_multiplier=1.0, round_nearest=8, dropout=0.2):
        super().__init__()

        input_dim = _make_divisible(32 * width_multiplier, round_nearest)
        last_dim = _make_divisible(1280 * width_multiplier, round_nearest)

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1,  16, 1, 1],
                [6,  24, 2, 2],
                [6,  32, 3, 2],
                [6,  64, 4, 2],
                [6,  96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]
            ]
        assert len(inverted_residual_setting) > 0 and len(inverted_residual_setting[0]) == 4

        features = [
            *conv_norm_activation(3, input_dim, 3, 2)
        ]

        for t, c, n, s in inverted_residual_setting:
            output_dim = _make_divisible(c * width_multiplier, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_dim, output_dim, stride, t))
                input_dim = output_dim

        features.extend(
            *conv_norm_activation(input_dim, last_dim, 1)
        )

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Linear(last_dim, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
