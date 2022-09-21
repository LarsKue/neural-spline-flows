
import torch

from .binned import BinnedSpline


class LinearSpline(BinnedSpline):
    def __init__(self, *args, bins: int = 10, **kwargs):
        #       parameter                           constraints             count
        # 1.    the domain half-width B             positive                1
        # 2.    the relative width of each bin      positive, sum to 1      #bins
        # 3.    the relative height of each bin     positive, sum to 1      #bins
        super().__init__(*args, **kwargs, bins=bins, parameter_counts=[1, bins, bins])

    def _spline1(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, bottom: torch.Tensor, top: torch.Tensor, *params: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return linear_spline(x, left, right, bottom, top, rev=rev)

    def _spline2(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, bottom: torch.Tensor, top: torch.Tensor, *params: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return linear_spline(x, left, right, bottom, top, rev=rev)


def linear_spline(x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, bottom: torch.Tensor, top: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    dx = right - left
    dy = top - bottom

    if not rev:
        xi = (x - left) / dx
        out = dy / dx * xi + bottom
    else:
        y = x

        xi = (y - bottom) * dx / dy
        out = xi * dx + left

    log_jac = torch.log(dy) - torch.log(dx)

    return out, log_jac
