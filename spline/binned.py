import torch
import torch.nn.functional as F
import numpy as np

from typing import Callable

from FrEIA.modules.coupling_layers import _BaseCouplingBlock

import utils


class BinnedSpline(_BaseCouplingBlock):
    """
    Base Class for Spline Couplings
    Splits Input into outside and inside a spline domain.
    The spline domain is further split into a fixed number of bins.
    Bin widths and heights are predicted by the subnetwork.
    The subnetwork may predict additional parameters for subclasses.
    Splining is performed by the subclass, based on the left, right, bottom and top edges for each bin.
    Input outside the spline domain is unaltered.
    """

    def __init__(self, dims_in, dims_c=None, subnet_constructor: Callable = None, split_len: float | int = 0.5,
                 bins: int = 10, parameter_counts: list[int, ...] = None):
        if dims_c is None:
            dims_c = []

        super().__init__(dims_in, dims_c, clamp=0.0, clamp_activation=lambda u: u, split_len=split_len)

        num_params = sum(parameter_counts)
        self.subnet1 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * num_params)
        self.subnet2 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * num_params)

        self.bins = bins
        self.parameter_counts = parameter_counts

    def _spline1(self,
                 x: torch.Tensor,
                 left: torch.Tensor,
                 right: torch.Tensor,
                 bottom: torch.Tensor,
                 top: torch.Tensor,
                 *params: torch.Tensor,
                 rev: bool = False,
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the spline for input x within a bin with edges left, right, bottom, top
        """
        raise NotImplementedError

    def _spline2(self,
                 x: torch.Tensor,
                 left: torch.Tensor,
                 right: torch.Tensor,
                 bottom: torch.Tensor,
                 top: torch.Tensor,
                 *params: torch.Tensor,
                 rev: bool = False,
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _split_params1(self, unconstrained_params: torch.Tensor) -> list[torch.Tensor]:
        unconstrained_params = unconstrained_params.movedim(1, -1)

        # move and split the parameter dimension, this simplifies some transformations
        unconstrained_params = unconstrained_params.reshape(*unconstrained_params.shape[:-1], self.split_len1, -1)

        return torch.split(unconstrained_params, self.parameter_counts, dim=-1)

    def _split_params2(self, unconstrained_params: torch.Tensor) -> list[torch.Tensor]:
        unconstrained_params = unconstrained_params.movedim(1, -1)

        # move and split the parameter dimension, this simplifies some transformations
        unconstrained_params = unconstrained_params.reshape(*unconstrained_params.shape[:-1], self.split_len2, -1)

        return torch.split(unconstrained_params, self.parameter_counts, dim=-1)

    def _constrain_params1(self, *params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        domain_width, bin_widths, bin_heights, *params = params

        # constrain the domain width to positive values
        # use a shifted softplus
        shift = np.log(np.e - 1)
        domain_width = F.softplus(domain_width + shift)

        # bin widths must be positive and sum to 1
        bin_widths = F.softmax(bin_widths, dim=-1)

        # same for the bin heights
        bin_heights = F.softmax(bin_heights, dim=-1)

        return domain_width, bin_widths, bin_heights, *params

    def _constrain_params2(self, *params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        domain_width, bin_widths, bin_heights, *params = params

        # constrain the domain width to positive values
        # use a shifted softplus
        shift = np.log(np.e - 1)
        domain_width = F.softplus(domain_width + shift)

        # bin widths must be positive and sum to 1
        bin_widths = F.softmax(bin_widths, dim=-1)

        # same for the bin heights
        bin_heights = F.softmax(bin_heights, dim=-1)

        return domain_width, bin_widths, bin_heights, *params

    def _coupling1(self, x1: torch.Tensor, u2: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        unconstrained_params = self.subnet1(u2)
        split_params = self._split_params1(unconstrained_params)
        domain_width, bin_widths, bin_heights, *params = self._constrain_params1(*split_params)

        return binned_spline(x1, domain_width=domain_width, bin_widths=bin_widths, bin_heights=bin_heights,
                             spline=self._spline1, spline_params=params, rev=rev)

    def _coupling2(self, x2: torch.Tensor, u1: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        unconstrained_params = self.subnet2(u1)
        split_params = self._split_params2(unconstrained_params)
        domain_width, bin_widths, bin_heights, *params = self._constrain_params2(*split_params)

        return binned_spline(x2, domain_width=domain_width, bin_widths=bin_widths, bin_heights=bin_heights,
                             spline=self._spline2, spline_params=params, rev=rev)


def make_knots(domain_width: torch.Tensor,
               bin_widths: torch.Tensor,
               bin_heights: torch.Tensor,
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find bin knots (x and y coordinates) from constrained parameters
    :param domain_width: half-width of the zero-centered spline box
    :param bin_widths: relative widths of each bin
    :param bin_heights: relative heights of each bin
    :return: tuple containing bin knot x and y coordinates
    """

    xs = torch.cumsum(bin_widths, dim=-1)
    pad = xs.new_zeros((*xs.shape[:-1], 1))
    xs = torch.cat((pad, xs), dim=-1)
    xs = 2 * domain_width * xs - domain_width

    ys = torch.cumsum(bin_heights, dim=-1)
    pad = ys.new_zeros((*ys.shape[:-1], 1))
    ys = torch.cat((pad, ys), dim=-1)
    ys = 2 * domain_width * ys - domain_width

    return xs, ys


def make_edges(x: torch.Tensor,
               xs: torch.Tensor,
               ys: torch.Tensor,
               rev: bool = False,
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def binned_spline(x: torch.Tensor,
                  *,
                  domain_width: torch.Tensor,
                  bin_widths: torch.Tensor,
                  bin_heights: torch.Tensor,
                  spline: Callable,
                  spline_params: tuple = (),
                  rev: bool = False,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = make_knots(domain_width, bin_widths, bin_heights)

    if not rev:
        inside = (xs[..., 0] < x) & (x <= xs[..., 1])
    else:
        y = x
        inside = (ys[..., 0] < y) & (y <= ys[..., 1])

    left, right, bottom, top = make_edges(x[inside], xs[inside], ys[inside], rev=rev)

    spline_out, spline_log_jac = spline(x, left, right, bottom, top, *spline_params, rev=rev)

    # identity tails
    out = torch.clone(x)
    # overwrite inside with spline
    out[inside] = spline_out
    # same for jacobian; logjac of identity is zero
    log_jac = out.new_zeros(out.shape)
    log_jac[inside] = spline_log_jac

    log_jac_det = utils.sum_except_batch(log_jac)

    if rev:
        log_jac_det = -log_jac_det

    return out, log_jac_det
