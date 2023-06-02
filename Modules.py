import torch
from torch import Tensor
from torch.nn import Sequential, Conv2d, MaxPool2d, BatchNorm1d, BatchNorm2d, Linear, ReLU, Flatten, Module, \
    Dropout1d, Dropout2d

from typing import List, Optional, Tuple


class ConvNetArgs:

    conv_params: List[dict]
    fc_params: List[dict]

    input_shape: Optional[Tuple[int, int, int]]
    output_shape: int

    def to_dict(self) -> dict:
        return dict(conv_params=self.conv_params, fc_params=self.fc_params)


def ConvBlock(channels: List[int], ksp: List[int],
              batch_norm: Optional[bool] = False,
              activation: Optional[Module] = None,
              dropout: Optional[bool] = False, dropout_p: Optional[float] = 0.5,
              pool: Optional[Module] = None, pool_ksp: Optional[List[int]] = None) -> Sequential:

    layers = [Conv2d(channels[0], channels[1], kernel_size=ksp[0], stride=ksp[1], padding=ksp[2])]

    if batch_norm:
        layers.append(BatchNorm2d(channels[1]))
    if activation:
        layers.append(activation())
    if dropout:
        layers.append(Dropout2d(p=dropout_p))
    if pool:
        layers.append(pool(kernel_size=pool_ksp[0], stride=pool_ksp[1], padding=pool_ksp[2]))

    return Sequential(*layers)


def FCBlock(dims: List[int], bias: Optional[bool] = True,
            batch_norm: Optional[bool] = False,
            dropout: Optional[Module] = None, dropout_p: Optional[float] = 0.5,
            activation: Optional = None) -> Sequential:

    layers = [Linear(dims[0], dims[1], bias=bias)]

    if batch_norm:
        layers.append(BatchNorm1d(dims[1]))
    if activation:
        layers.append(activation())
    if dropout:
        layers.append(Dropout1d(p=dropout_p))

    return Sequential(*layers)


class ConvNet(Module):

    def __init__(self, module_args: ConvNetArgs):

        super(ConvNet, self).__init__()

        if module_args.conv_params[0]["channels"][0] is None:
            assert module_args.input_shape is not None
            module_args.conv_params[0]["channels"][0] = module_args.input_shape[0]
        if module_args.fc_params[-1]["dims"][1] is None:
            assert module_args.output_shape is not None
            module_args.fc_params[-1]["dims"][1] = module_args.output_shape

        self.conv_layers = Sequential(*(ConvBlock(**cp) for cp in module_args.conv_params))
        if module_args.fc_params[0]["dims"][0] is None:
            assert module_args.input_shape is not None
            module_args.fc_params[0]["dims"][0] = \
                self.conv_layers(torch.zeros((2, ) + module_args.input_shape)).shape[1:].numel()

        self.flatten_layer = Flatten()
        self.fc_layers = Sequential(*(FCBlock(**fp) for fp in module_args.fc_params))

        print("Created ConvNet with params: ")
        print(self)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc_layers(self.flatten_layer(self.conv_layers(x)))


def LeNet_params(input_shape: Tuple[int, int, int], output_shape: int) -> ConvNetArgs:

    lenet_args = ConvNetArgs()
    lenet_args.conv_params = [dict(channels=[None, 6], ksp=[5, 1, 0],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.05,
                                   pool=MaxPool2d, pool_ksp=[2, 2, 0]),
                              dict(channels=[6, 16], ksp=[5, 1, 0],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.075,
                                   pool=MaxPool2d, pool_ksp=[2, 2, 0])]
    lenet_args.fc_params = [dict(dims=[None, 84], bias=True,
                                 batch_norm=True,
                                 dropout=True, dropout_p=0.15,
                                 activation=ReLU),
                            dict(dims=[84, None], bias=True,
                                 batch_norm=False,
                                 activation=None)]

    lenet_args.input_shape = input_shape
    lenet_args.output_shape = output_shape

    return lenet_args


def Conv4_params(input_shape: Tuple[int, int, int], output_shape: int) -> ConvNetArgs:

    conv4_args = ConvNetArgs()
    conv4_args.conv_params = [dict(channels=[None, 64], ksp=[3, 1, 1],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.1),
                              dict(channels=[64, 64], ksp=[3, 1, 1],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.1,
                                   pool=MaxPool2d, pool_ksp=[2, 2, 0]),
                              dict(channels=[64, 128], ksp=[3, 1, 1],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.15),
                              dict(channels=[128, 128], ksp=[3, 1, 1],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.15,
                                   pool=MaxPool2d, pool_ksp=[2, 2, 0])]
    conv4_args.fc_params = [dict(dims=[None, 256], bias=True,
                                 batch_norm=True,
                                 dropout=True, dropout_p=0.25,
                                 activation=ReLU),
                            dict(dims=[256, 256], bias=True,
                                 batch_norm=True,
                                 dropout=True, dropout_p=0.25,
                                 activation=ReLU),
                            dict(dims=[256, None], bias=True,
                                 batch_norm=False,
                                 activation=None)]

    conv4_args.input_shape = input_shape
    conv4_args.output_shape = output_shape

    return conv4_args


def Conv6_params(input_shape: Tuple[int, int, int], output_shape: int) -> ConvNetArgs:

    conv6_args = ConvNetArgs()
    conv6_args.conv_params = [dict(channels=[None, 64], ksp=[3, 1, 1],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.1),
                              dict(channels=[64, 64], ksp=[3, 1, 1],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.1,
                                   pool=MaxPool2d, pool_ksp=[2, 2, 0]),
                              dict(channels=[64, 128], ksp=[3, 1, 1],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.15),
                              dict(channels=[128, 128], ksp=[3, 1, 1],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.15,
                                   pool=MaxPool2d, pool_ksp=[2, 2, 0]),
                              dict(channels=[128, 256], ksp=[3, 1, 1],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.2),
                              dict(channels=[256, 256], ksp=[3, 1, 1],
                                   batch_norm=True,
                                   activation=ReLU,
                                   dropout=True, dropout_p=0.2,
                                   pool=MaxPool2d, pool_ksp=[2, 2, 0])]
    conv6_args.fc_params = [dict(dims=[None, 256], bias=True,
                                 batch_norm=True,
                                 dropout=True, dropout_p=0.25,
                                 activation=ReLU),
                            dict(dims=[256, 256], bias=True,
                                 batch_norm=True,
                                 dropout=True, dropout_p=0.25,
                                 activation=ReLU),
                            dict(dims=[256, None], bias=True,
                                 batch_norm=False,
                                 activation=None)]

    conv6_args.input_shape = input_shape
    conv6_args.output_shape = output_shape

    return conv6_args
    