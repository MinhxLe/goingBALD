import torch.nn as nn
from typing import List

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int, # input n layers
            out_channels: int, # output n layers
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            add_residual: bool = False):
        super().__init__()
        self.activate = nn.ReLU()
        self.add_residual = add_residual
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        # adding residual unit
        self.down_sample = None
        if add_residual and in_channels != out_channels:
            # kernel 1 is identity
            self.down_sample = nn.Conv1d(in_channels, out_channels, 1)
        self.init_weights()

    def init_weights(self):
        # TODO understand kaiming initialization
        nn.init.kaiming_uniform_(
            self.conv.weight.data, mode='fan_in', nonlinearity='relu')
        if self.conv.bias is not None:
            self.conv.bias.data.fill_(0)
        if self.down_sample is not None:
            nn.init.kaiming_uniform_(
                self.down_sample.weight.data,
                mode='fan_in', nonlinearity='relu')
            if self.down_sample.bias is not None:
                self.down_sample.bias.data.fill_(0)

    def forward(self, inputs):
        # input.shape (batchsize, in_channels, seq_len)
        # output.shape (batchsize, out_channels, out seq_len)
        output = self.activate(self.conv(inputs))
        if self.add_residual:
            output += self.down_sample(inputs) if self.down_sample else inputs
        return output


class ConvNet(nn.Module):
    """
    CNN that doesn't  change shape of output except # of layers
    """
    def __init__(
            self,
            channels: List[int],
            kernel_size: int=3,
            dropout: float=0.5,
            is_dilated: bool=False,
            add_residual: bool=True):
        super().__init__()
        layers = []
        n_layers = len(channels)-1
        for i in range(n_layers):
            in_channels = channels[i]
            out_channels = channels[i+1]
            # TODO understand why we have padding
            dilation = kernel_size ** i if is_dilated else 1
            # add padding based on kernel size and dilation
            padding = (kernel_size -1)//2 * dilation
            layers += [
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding, dilation=dilation,
                    add_residual=add_residual),
                nn.Dropout(dropout),
            ]
        # not including last dropout layer
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, inputs):
        return self.net(inputs)
