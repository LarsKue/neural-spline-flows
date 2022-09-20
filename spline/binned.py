
import torch
from typing import Callable

from FrEIA.modules.coupling_layers import _BaseCouplingBlock


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

    def _spline1(self, x: torch.Tensor, x_left: torch.Tensor, x_right: torch.Tensor, y_left: torch.Tensor, y_right: torch.Tensor, *params: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _spline2(self, x: torch.Tensor, x_left: torch.Tensor, x_right: torch.Tensor, y_left: torch.Tensor, y_right: torch.Tensor, *params: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _get_bin_edges(self, x, xs, ys, rev=False):
        # find left and right bin edge indices
        if not rev:
            right = torch.searchsorted(xs, x[..., None])
            left = right - 1
        else:
            y = x
            right = torch.searchsorted(ys, y[..., None])
            left = right - 1

        # get left and right bin edge values
        # variables are named as in the paper
        # e.g. xk is $x^{(k)}$ and xkp is $x^{(k+1)}$
        x_left = torch.gather(xs, dim=-1, index=left).squeeze(-1)
        x_right = torch.gather(xs, dim=-1, index=right).squeeze(-1)
        y_left = torch.gather(ys, dim=-1, index=left).squeeze(-1)
        y_right = torch.gather(ys, dim=-1, index=right).squeeze(-1)

        return x_left, x_right, y_left, y_right

    def _coupling1(self, x1, u2, rev=False):
        params = self.subnet1(u2)
        split_sizes = self.split_len1 * torch.as_tensor(self.parameter_counts, dtype=torch.int64)
        xs, ys, *params = torch.split(params, split_sizes.tolist(), dim=1)

        x_left, x_right, y_left, y_right = self._get_bin_edges(x1, xs, ys, rev=rev)

        return self._spline1(x1, x_left, x_right, y_left, y_right, *params, rev=rev)

    def _coupling2(self, x2, u1, rev=False):
        params = self.subnet2(u1)
        split_sizes = self.split_len2 * torch.as_tensor(self.parameter_counts, dtype=torch.int64)
        xs, ys, *params = torch.split(params, split_sizes.tolist(), dim=1)

        x_left, x_right, y_left, y_right = self._get_bin_edges(x2, xs, ys, rev=rev)

        return self._spline2(x2, x_left, x_right, y_left, y_right, *params, rev=rev)

