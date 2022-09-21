
import torch
import torch.nn.functional as F
import numpy as np

from typing import Callable

from FrEIA.modules.coupling_layers import _BaseCouplingBlock

import utils


class BinnedSpline(_BaseCouplingBlock):
    def __init__(self, dims_in, dims_c=None, subnet_constructor: Callable = None, split_len: float | int = 0.5, bins: int = 10, parameter_counts: list[int, ...] = None):
        if dims_c is None:
            dims_c = []

        super().__init__(dims_in, dims_c, clamp=0.0, clamp_activation=lambda u: u, split_len=split_len)

        num_params = sum(parameter_counts)
        self.subnet1 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * num_params)
        self.subnet2 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * num_params)

        self.bins = bins
        self.parameter_counts = parameter_counts

    def _spline1(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, bottom: torch.Tensor, top: torch.Tensor, *params: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _spline2(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, bottom: torch.Tensor, top: torch.Tensor, *params: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _coupling1(self, x1: torch.Tensor, u2: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        unconstrained_params = self.subnet1(u2)

        # TODO: make tailed spline a base case and binned spline an inner variant of the tailed spline
        #  tailed spline technically only needs an inner spline and a domain box
        #  tails can then be affine
        #  inner spline can get domain box from tailed spline

        xs, ys = make_knots(unconstrained_params, bins=self.bins, split_len=self.split_len1)

        if not rev:
            inside = (xs[..., 0] < x1) & (x1 <= xs[..., 1])
        else:
            y1 = x1
            inside = (ys[..., 0] < y1) & (y1 <= ys[..., 1])

        left, right, bottom, top = make_edges(x1[inside], xs[inside], ys[inside], rev=rev)

        spline_out, spline_log_jac = self._spline1(x1, left, right, bottom, top, rev=rev)

        # identity tails
        out = torch.clone(x1)
        # overwrite inside with spline
        out[inside] = spline_out
        # same for jacobian; logjac of identity is zero
        log_jac = out.new_zeros(out.shape)
        log_jac[inside] = spline_log_jac

        log_jac_det = utils.sum_except_batch(log_jac)

        if rev:
            log_jac_det = -log_jac_det

        return out, log_jac_det

    def _coupling2(self, x2: torch.Tensor, u1: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        unconstrained_params = self.subnet2(u1)

        xs, ys = make_knots(unconstrained_params, bins=self.bins, split_len=self.split_len2)

        if not rev:
            inside = (xs[..., 0] < x2) & (x2 <= xs[..., 1])
        else:
            y2 = x2
            inside = (ys[..., 0] < y2) & (y2 <= ys[..., 1])

        left, right, bottom, top = make_edges(x2[inside], xs[inside], ys[inside], rev=rev)

        spline_out, spline_log_jac = self._spline2(x2, left, right, bottom, top, rev=rev)

        # identity tails
        out = torch.clone(x2)
        # overwrite inside with spline
        out[inside] = spline_out
        # same for jacobian; logjac of identity is zero
        log_jac = out.new_zeros(out.shape)
        log_jac[inside] = spline_log_jac

        log_jac_det = utils.sum_except_batch(log_jac)

        if rev:
            log_jac_det = -log_jac_det

        return out, log_jac_det


def make_knots(unconstrained_params: torch.Tensor, bins: int, split_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find bin knots (x and y coordinates) from unconstrained parameter outputs
    :param unconstrained_params: unconstrained subnetwork outputs (domain half-width, bin widths, bin heights, bin deltas)
    :param bins: number of bins to use
    :param split_len: split length used by the current coupling
    :return: tuple containing bin knot x and y coordinates
    """
    # move and split the parameter dimension, this simplifies some transformations
    unconstrained_params = unconstrained_params.movedim(1, -1)
    unconstrained_params = unconstrained_params.reshape(*unconstrained_params.shape[:-1], split_len, 2 * bins + 1)

    domain_width, bin_widths, bin_heights = torch.split(unconstrained_params, [1, bins, bins], dim=-1)

    # constrain the domain width to positive values
    # use a shifted softplus
    shift = np.log(np.e - 1)
    domain_width = F.softplus(domain_width + shift)

    # bin widths must be positive and sum to 1
    bin_widths = F.softmax(bin_widths, dim=-1)
    xs = torch.cumsum(bin_widths, dim=-1)
    pad = xs.new_zeros((*xs.shape[:-1], 1))
    xs = torch.cat((pad, xs), dim=-1)
    xs = 2 * domain_width * xs - domain_width

    # same for the bin heights
    bin_heights = F.softmax(bin_heights, dim=-1)
    ys = torch.cumsum(bin_heights, dim=-1)
    pad = ys.new_zeros((*ys.shape[:-1], 1))
    ys = torch.cat((pad, ys), dim=-1)
    ys = 2 * domain_width * ys - domain_width

    return xs, ys


def make_edges(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find bin edges for the given input values and bin knots
    :param x: input tensor of shape (batch_size, ...)
    :param xs: bin knot x coordinates of shape (batch_size, ..., bins)
    :param ys: bin knot y coordinates of shape (batch_size, ..., bins)
    :param rev: whether to run in reverse
    :return: tuple containing left, right, bottom and top bin edges
    """
    # find upper and lower bin edge indices
    if not rev:
        upper = torch.searchsorted(xs, x[..., None])
        lower = upper - 1
    else:
        y = x
        upper = torch.searchsorted(ys, y[..., None])
        lower = upper - 1

    left_edge = torch.gather(xs, dim=-1, index=lower).squeeze(-1)
    right_edge = torch.gather(xs, dim=-1, index=upper).squeeze(-1)
    bottom_edge = torch.gather(ys, dim=-1, index=lower).squeeze(-1)
    top_edge = torch.gather(ys, dim=-1, index=upper).squeeze(-1)

    return left_edge, right_edge, bottom_edge, top_edge
