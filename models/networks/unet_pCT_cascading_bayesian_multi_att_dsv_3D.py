import torch.nn as nn
import torch
from .utils import UnetConv3, UnetUp3_CT, UnetGridGatingSignal3, UnetDsv3, ResidualBlock3d
import torch.nn.functional as F
from torchvision.transforms import Resize
from models.networks_other import init_weights
from models.layers.grid_attention_layer import GridAttentionBlock3D


class unet_pCT_cascading_bayesian_multi_att_dsv_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=2, is_deconv=True, in_channels=4, prior_information_channels=0,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True, conv_bloc_type=None,
                 bayesian_skip_type='conv'):
        super(unet_pCT_cascading_bayesian_multi_att_dsv_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.prior_information_channels = prior_information_channels
        self.bayesian_skip_type = bayesian_skip_type
        conv_bloc_class = UnetConv3
        if conv_bloc_type is not None:
            if conv_bloc_type == 'classic':
                conv_bloc_class = UnetConv3
            if conv_bloc_type == 'residual':
                conv_bloc_class = ResidualBlock3d

        if self.bayesian_skip_type not in ['conv', 'add']:
            raise NotImplementedError(f'{self.bayesian_skip_type} is not implemented, use one of [\'conv\', \'add\']')
        if self.bayesian_skip_type == 'add':
            assert len(self.prior_information_channels) == n_classes, \
                'Prior information needs to have the same number of channels als final output. Maybe one hot encode it?'

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_bloc_class(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.prior_resize1 = nn.Upsample(scale_factor=0.5)

        self.conv2 = conv_bloc_class(filters[0] + 1, filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.prior_resize2 = nn.Upsample(scale_factor=0.5)

        self.conv3 = conv_bloc_class(filters[1] + 1, filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.prior_resize3 = nn.Upsample(scale_factor=0.5)

        self.conv4 = conv_bloc_class(filters[2] + 1, filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.prior_resize4 = nn.Upsample(scale_factor=0.5)

        self.center = conv_bloc_class(filters[3] + 1, filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4] + 1, filters[3], is_batchnorm, conv_bloc_class=conv_bloc_class)
        self.up_concat3 = UnetUp3_CT(filters[3] + 1, filters[2], is_batchnorm, conv_bloc_class=conv_bloc_class)
        self.up_concat2 = UnetUp3_CT(filters[2] + 1, filters[1], is_batchnorm, conv_bloc_class=conv_bloc_class)
        self.up_concat1 = UnetUp3_CT(filters[1] + 1, filters[0], is_batchnorm, conv_bloc_class=conv_bloc_class)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(n_classes*4, n_classes, 1)

        if self.bayesian_skip_type == 'conv':
            # let prior information skip whole network and integrate it here
            self.final_bayesian_skip = nn.Conv3d(n_classes + len(self.prior_information_channels), n_classes, 1)

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
        prior_resize1 = self.prior_resize1(inputs[:, self.prior_information_channels])

        conv2 = self.conv2(torch.cat([maxpool1, prior_resize1], dim=1))
        maxpool2 = self.maxpool2(conv2)
        prior_resize2 = self.prior_resize2(prior_resize1)

        conv3 = self.conv3(torch.cat([maxpool2, prior_resize2], dim=1))
        maxpool3 = self.maxpool3(conv3)
        prior_resize3 = self.prior_resize3(prior_resize2)

        conv4 = self.conv4(torch.cat([maxpool3, prior_resize3], dim=1))
        maxpool4 = self.maxpool4(conv4)
        prior_resize4 = self.prior_resize3(prior_resize3)

        # Gating Signal Generation
        center = self.center(torch.cat([maxpool4, prior_resize4], dim=1))
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, torch.cat([center, prior_resize4], dim=1))
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, torch.cat([up4, prior_resize3], dim=1))
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, torch.cat([up3, prior_resize2], dim=1))
        up1 = self.up_concat1(conv1, torch.cat([up2, prior_resize1], dim=1))

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

        # Bayesian skip connection
        if self.bayesian_skip_type == 'conv':
            final_with_bayesian_skip = self.final_bayesian_skip(
                torch.cat([inputs[:, self.prior_information_channels], final], dim = 1))
            return final_with_bayesian_skip

        elif self.bayesian_skip_type == 'add':
            final += inputs[:, self.prior_information_channels]
            return final


    @staticmethod
    def apply_argmax_softmax(pred, dim=1):
        if dim is None:
            log_p = F.sigmoid(pred)
        else:
            log_p = F.softmax(pred, dim=dim)

        return log_p


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)


