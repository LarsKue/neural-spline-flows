
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from FrEIA.modules.coupling_layers import _BaseCouplingBlock

import utils


class RationalQuadraticSpline(_BaseCouplingBlock):
    """
    Rational Quadratic Spline Coupling as described in arXiv:1906.04032
    """
    def __init__(self, dims_in, dims_c=None, subnet_constructor: Callable = None, split_len: float | int = 0.5, bins: int = 10):
        if dims_c is None:
            dims_c = []
        super().__init__(dims_in, dims_c, clamp=0.0, clamp_activation=lambda u: u, split_len=split_len)

        # for each input we have the following parameters (constraints):
        # 1. the domain width B (positive)
        # 2. the relative width of each bin (positive, sum to 1)
        # 3. the relative height of each bin (positive, sum to 1)
        # 4. the derivative at the edge of each inner bin (positive)

        # so in total, there are 1 + K + K + (K - 1) = 3 * K parameters for each input value
        # where K is the number of bins
        self.subnet1 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * 3 * bins)
        self.subnet2 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * 3 * bins)

        self.bins = bins

    def transform_params(self, params, split_len):
        if torch.any(torch.isnan(params)):
            # raise this error here because otherwise this raises an IndexError later which is cryptic
            sum_nans = torch.sum(torch.isnan(params))
            raise ValueError(f"Cannot transform params because they contain {sum_nans} NaN values.")

        split_sizes = [
            split_len,
            split_len * self.bins,
            split_len * self.bins,
            split_len * (self.bins - 1)
        ]
        B, widths, heights, deltas = torch.split(params, split_sizes, dim=1)

        batch_size = params.shape[0]

        # split parameters from channels and move them to the last dimension
        # this makes following transformations much simpler
        B = B.reshape(batch_size, 1, split_len, *B.shape[2:]).movedim(1, -1)
        widths = widths.reshape(batch_size, self.bins, split_len, *widths.shape[2:]).movedim(1, -1)
        heights = heights.reshape(batch_size, self.bins, split_len, *heights.shape[2:]).movedim(1, -1)
        deltas = deltas.reshape(batch_size, self.bins - 1, split_len, *deltas.shape[2:]).movedim(1, -1)

        # define activation for B such that if the network predicts 0, we get B = 1
        # we use a shifted softplus
        B = F.softplus(B + np.log(np.e - 1))

        widths = F.softmax(widths, dim=-1)
        heights = F.softmax(heights, dim=-1)

        # cumulative sum of relative widths, starting with zeros
        xs = torch.cumsum(widths, dim=-1)
        pad = xs.new_zeros((*xs.shape[:-1], 1))
        xs = torch.cat((pad, xs), dim=-1)

        xs = 2 * B * xs - B

        # do the same for heights
        ys = torch.cumsum(heights, dim=-1)
        pad = ys.new_zeros((*ys.shape[:-1], 1))
        ys = torch.cat((pad, ys), dim=-1)

        ys = 2 * B * ys - B

        # shifted softplus for network 0 -> delta = 1
        deltas = F.softplus(deltas + np.log(np.e - 1))

        # add tails
        pad = deltas.new_ones((*deltas.shape[:-1], 2))
        deltas = torch.cat((deltas, pad), dim=-1).roll(1, dims=-1)

        # avoid warnings in the coupling
        xs = xs.contiguous()
        ys = ys.contiguous()
        deltas = deltas.contiguous()

        return xs, ys, deltas

    def _coupling(self, x: torch.Tensor, u: torch.Tensor, rev: bool = False, *, subnet: nn.Module, split_len: int):
        params = subnet(u)

        xs, ys, deltas = self.transform_params(params, split_len)

        inside = (xs[..., 0] < x) & (x <= xs[..., -1])

        spline_out, spline_log_jac = self._spline(x[inside], xs[inside], ys[inside], deltas[inside], rev=rev)

        out = torch.clone(x)
        out[inside] = spline_out
        log_jac = out.new_zeros(out.shape)
        log_jac[inside] = spline_log_jac

        log_jac_det = utils.sum_except_batch(log_jac)

        if rev:
            log_jac_det = -log_jac_det

        return out, log_jac_det

    def _spline(self, x, xs, ys, deltas, rev=False):
        # find left and right bin edge indices
        if rev:
            y = x
            right = torch.searchsorted(ys, y[..., None])
            left = right - 1
        else:
            right = torch.searchsorted(xs, x[..., None])
            left = right - 1

        # get left and right bin edge values
        # variables are named as in the paper
        # e.g. xk is $x^{(k)}$ and xkp is $x^{(k+1)}$
        xk = torch.gather(xs, dim=-1, index=left).squeeze(-1)
        xkp = torch.gather(xs, dim=-1, index=right).squeeze(-1)
        yk = torch.gather(ys, dim=-1, index=left).squeeze(-1)
        ykp = torch.gather(ys, dim=-1, index=right).squeeze(-1)
        dk = torch.gather(deltas, dim=-1, index=left).squeeze(-1)
        dkp = torch.gather(deltas, dim=-1, index=right).squeeze(-1)

        # define some commonly used values
        dx = xkp - xk
        dy = ykp - yk
        sk = dy / dx

        if not rev:
            xi = (x - xk) / dx

            # Eq 4 in the paper
            numerator = dy * (sk * xi ** 2 + dk * xi * (1 - xi))
            denominator = sk + (dkp + dk - 2 * sk) * xi * (1 - xi)
            out = yk + numerator / denominator
        else:
            y = x
            # Eq 6-8 in the paper
            a = dy * (sk - dk) + (y - yk) * (dkp + dk - 2 * sk)
            b = dy * dk - (y - yk) * (dkp + dk - 2 * sk)
            c = -sk * (y - yk)

            # Eq 29 in the appendix of the paper
            discriminant = b ** 2 - 4 * a * c
            assert torch.all(discriminant >= 0)

            xi = 2 * c / (-b - torch.sqrt(discriminant))

            out = xi * dx + xk

        # Eq 5 in the paper
        numerator = sk ** 2 * (dkp * xi ** 2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
        denominator = (sk + (dkp + dk - 2 * sk) * xi * (1 - xi)) ** 2
        log_jac = torch.log(numerator) - torch.log(denominator)

        return out, log_jac

    def _coupling1(self, x1, u2, rev=False):
        return self._coupling(x1, u2, rev=rev, subnet=self.subnet1, split_len=self.split_len1)

    def _coupling2(self, x2, u1, rev=False):
        return self._coupling(x2, u1, rev=rev, subnet=self.subnet2, split_len=self.split_len2)