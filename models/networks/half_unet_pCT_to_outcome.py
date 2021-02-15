import torch.nn as nn
import torch
from .utils import UnetConv3, UnetUp3_CT, UnetGridGatingSignal3, UnetDsv3, ResidualBlock3d
import torch.nn.functional as F
from models.networks_other import init_weights
from models.layers.grid_attention_layer import GridAttentionBlock3D


class half_unet_pCT_to_outcome(nn.Module):

    def __init__(self, feature_scale=4, n_classes=2, is_deconv=True, in_channels=4,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True, conv_bloc_type=None):
        super(half_unet_pCT_to_outcome, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        conv_bloc_class = UnetConv3
        if conv_bloc_type is not None:
            if conv_bloc_type == 'classic':
                conv_bloc_class = UnetConv3
            if conv_bloc_type == 'residual':
                conv_bloc_class = ResidualBlock3d


        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_bloc_class(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = conv_bloc_class(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = conv_bloc_class(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = conv_bloc_class(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = conv_bloc_class(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool5 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # final conv (without any concat)
        self.final = nn.Conv3d(n_classes, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        maxpool5 = self.maxpool5(center)

        final = self.final(maxpool5)

        return final


    @staticmethod
    def apply_argmax_softmax(pred, dim=1):
        if dim is None:
            log_p = F.sigmoid(pred)
        else:
            log_p = F.softmax(pred, dim=dim)

        return log_p

