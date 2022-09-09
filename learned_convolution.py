
import FrEIA.modules as fm

import torch
import torch.nn as nn
from torch.nn.functional import conv2d


class LearnedConv1x1(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None):
        super().__init__(dims_in, dims_c)

        channels = next(iter(dims_in))[0]

        # initialize weight as a random orthogonal matrix
        # since this has logdet 0
        qr = torch.randn(channels, channels)
        weight = torch.linalg.qr(qr).Q
        self.weight = nn.Parameter(weight)

    def output_dims(self, input_dims):
        return input_dims

    def forward(self, x_or_z, c=None, rev=False, jac=True):
        if rev:
            weight = torch.linalg.inv(self.weight)
        else:
            weight = self.weight

        x = next(iter(x_or_z))
        height, width = x.shape[2:]
        logabsdet = height * width * torch.slogdet(weight)[1]
        weight = weight.reshape(*weight.shape, 1, 1)
        x = conv2d(x, weight)

        if rev:
            logabsdet = -logabsdet

        return (x,), logabsdet
