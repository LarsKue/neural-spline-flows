
import FrEIA.framework as ff
import FrEIA.modules as fm

import torch
import torch.nn as nn


class AdditiveCoupling(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, subnet_constructor=None):
        super().__init__(dims_in, dims_c)

        channels = next(iter(dims_in))[0]
        self.split_len1 = int(round(channels / 2))
        self.split_len2 = channels - self.split_len1

        self.subnet1 = subnet_constructor(self.split_len1, self.split_len2)
        self.subnet2 = subnet_constructor(self.split_len2, self.split_len1)

    def output_dims(self, input_dims):
        return input_dims

    def forward(self, x_or_z, c=None, rev=False, jac=True):
        if not rev:
            x = next(iter(x_or_z))

            x1, x2 = torch.chunk(x, chunks=2, dim=1)

            if c is None:
                z2 = x2 + self.subnet1(x1)
                z1 = x1 + self.subnet2(z2)
            else:
                c = next(iter(c))
                c1, c2 = torch.chunk(c, chunks=2, dim=1)
                x1c = torch.cat((x1, c1), dim=1)
                z2 = x2 + self.subnet1(x1c)
                z2c = torch.cat((z2, c2), dim=1)
                z1 = x1 + self.subnet2(z2c)

            z = torch.cat((z1, z2), dim=1)

            return (z,), 0.0
        else:
            z = next(iter(x_or_z))
            z1, z2 = torch.chunk(z, chunks=2, dim=1)

            if c is None:
                x1 = z1 - self.subnet2(z2)
                x2 = z2 - self.subnet1(x1)
            else:
                c = next(iter(c))
                c1, c2 = torch.chunk(c, chunks=2, dim=1)
                z2c = torch.cat((z2, c2), dim=1)
                x1 = z1 - self.subnet2(z2c)
                x1c = torch.cat((x1, c1), dim=1)
                x2 = z2 - self.subnet1(x1c)

            x = torch.cat((x1, x2), dim=1)

            return (x,), 0.0
