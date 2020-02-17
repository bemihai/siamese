"""
Implement OSNet as feature extractor.

Reference:
 - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
 - https://github.com/KaiyangZhou/deep-person-reid

Small changes from torchreid version:
 - remove instance normalization, grouped convolutions
 - keep only relu as gate activation
"""

from torch import nn
from torch.nn import functional as F

"""      Basic layers           """


class ConvLayer(nn.Module):
    """ Convolution layer: convolution + batch norm + relu. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """ 1x1 convolution + batch norm + relu """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """ 1x1 convolution + batch norm, without non-linearity """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """ 3x3 convolution + batch norm + relu """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LiteConv3x3(nn.Module):
    """
    Lite 3x3 convolution: 1x1 conv + depth-wise 3x3 conv + batch norm + relu
    """

    def __init__(self, in_channels, out_channels):
        super(LiteConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


"""    Building blocks for omni-scale feature learning   """


class ChannelGate(nn.Module):
    """ A mini-network that generates channel-wise gates conditioned on input tensor (AG) """

    def __init__(self, in_channels, n_gates=None, return_gates=False, reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        # TODO: what is number of gates ?
        if n_gates is None:
            n_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # TODO: reduction = AG hidden layer size ?
        mid_channels = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1, bias=True, padding=0)
        self.norm1 = None
        # TODO: how LayerNorm works ?
        if layer_norm:
            self.norm1 = nn.LayerNorm((mid_channels, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, n_gates, 1, bias=True, padding=0)
        self.gate_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        input_x = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input_x * x


class BaseBlock(nn.Module):
    """ Baseline bottleneck """

    def __init__(self, in_channels, out_channels, bottleneck_reduction=4, **kwargs):
        super(BaseBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = LiteConv3x3(mid_channels, mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class OSBlock(nn.Module):
    """ Omni-scale feature learning block (OSNet Bottleneck) """

    def __init__(self, in_channels, out_channels, bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        # multi-scale feature learners (exponent T = 4)
        self.conv2a = LiteConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LiteConv3x3(mid_channels, mid_channels),
            LiteConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LiteConv3x3(mid_channels, mid_channels),
            LiteConv3x3(mid_channels, mid_channels),
            LiteConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LiteConv3x3(mid_channels, mid_channels),
            LiteConv3x3(mid_channels, mid_channels),
            LiteConv3x3(mid_channels, mid_channels),
            LiteConv3x3(mid_channels, mid_channels),
        )
        # unified aggregation gate
        self.gate = ChannelGate(mid_channels)
        # fusion
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        # TODO: what is downsample doing ?
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


"""    OSNet Architecture       """


class OSNet(nn.Module):
    """
    Architecture of OSNet with input image size 256 x 128 (cf. Table 1 on page 5)
    """

    def __init__(
            self,
            blocks,                            # osnet bottlenecks
            layers,                            # TODO: nr of osnet blocks at each layer ?
            channels,                          # nr of output channels at each layer
            feature_dim=512,                   # learned features size
            **kwargs
    ):
        super(OSNet, self).__init__()
        n_blocks = len(blocks)
        assert n_blocks == len(layers)
        assert n_blocks == len(channels) - 1
        self.feature_dim = feature_dim

        # conv1: conv + max pooling
        # paper sizes: (3, 256, 128) -> (64, 128, 64) -> (64, 64, 32)
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # conv2: osnet bottlenecks + transition layer (1x1 conv + average pooling)
        # paper sizes: (64, 64, 32) -> (256, 64, 32) -> (256, 32, 16)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True
        )

        # conv3: osnet bottlenecks + transition layer (1x1 conv + average pooling)
        # paper sizes: (256, 32, 16) -> (384, 32, 15) -> (384, 16, 8)
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )

        # conv4: osnet bottlenecks, no transition layer (spatial dims are preserved)
        # paper sizes: (384, 16, 8) -> (512, 16, 8)
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )

        # conv5: 1x1 conv + average pooling to flatten the output
        # paper sizes: (512, 16, 8) -> (512, 16, 8) -> (512, 1, 1)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # fc: fully connected layer to get features
        self.fc = self._fc_layer(channels[3])

        # initializa learnable parameters
        self._init_params()

    @staticmethod
    def _make_layer(block, nr_blocks, in_channels, out_channels, reduce_spatial_size):
        """ stack osnet blocks together in a sequential layer"""
        layers = [block(in_channels, out_channels)]
        for i in range(1, nr_blocks):
            layers.append(block(out_channels, out_channels))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )
        return nn.Sequential(*layers)

    def _fc_layer(self, input_dim):
        """ fc layer to get multi-scale features """
        fc = nn.Sequential(
            nn.Linear(input_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True)
        )
        return fc

    def _init_params(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps=False):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


