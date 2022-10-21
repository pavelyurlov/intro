import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import MyResNet34Encoder


def conv_norm_activation(input_dim, output_dim, kernel_size, stride=1, padding=0, use_batchnorm=True):
    layers = [
        nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
    ]
    if use_batchnorm:
        layers.append(
            nn.BatchNorm2d(output_dim)
        )
    layers.append(
        nn.ReLU(inplace=True)
    )

    layers = nn.Sequential(*layers)
    return layers


class Attention(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim, input_dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // reduction, input_dim, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(input_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, skip_dim, output_dim, use_batchnorm=True, use_attention=False):
        super().__init__()

        self.conv1 = conv_norm_activation(input_dim + skip_dim, output_dim, 3, padding=1, use_batchnorm=use_batchnorm)
        self.attention1 = Attention(input_dim + skip_dim) if use_attention else None
        self.conv2 = conv_norm_activation(output_dim, output_dim, 3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = Attention(output_dim) if use_attention else None

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)

        return x


class UNetDecoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, n_blocks=5, use_batchnorm=True, use_attention=False):
        super().__init__()

        assert n_blocks == len(decoder_dim)

        encoder_dim = encoder_dim[1:][::-1]

        head_dim = encoder_dim[0]
        in_dim = [head_dim] + list(decoder_dim[:-1])
        skip_dim = list(encoder_dim[1:]) + [0]
        out_dim = decoder_dim

        blocks = [
            DecoderBlock(in_d, skip_d, out_d, use_batchnorm, use_attention)
            for in_d, skip_d, out_d in zip(in_dim, skip_dim, out_dim)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:][::-1]

        head = features[0]
        skips = features[1:]

        x = head
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_dim, out_dim, kernel_size=3, activation=None, upsampling=1):
        conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = activation if activation is not None else nn.Identity()
        super().__init__(conv, upsampling, activation)


class MyUNet(nn.Module):
    def __init__(self,
                 encoder='resnet34',
                 depth=5,
                 decoder_use_batchnorm=True,
                 decoder_use_attention=False,
                 decoder_channels=(256, 128, 64, 32, 16),
                 in_channels=3,
                 num_classes=1):
        super().__init__()

        assert depth == 5 and len(decoder_channels) == 5
        assert encoder == 'resnet34'

        self.encoder = MyResNet34Encoder(num_classes=num_classes)

        self.decoder = UNetDecoder(encoder_dim=self.encoder.out_channels,
                                   decoder_dim=decoder_channels,
                                   n_blocks=depth,
                                   use_batchnorm=decoder_use_batchnorm,
                                   use_attention=decoder_use_attention)

        self.segmentation_head = SegmentationHead(in_dim=decoder_channels[-1],
                                                  out_dim=num_classes)

        # initialise weights
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.segmentation_head.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
