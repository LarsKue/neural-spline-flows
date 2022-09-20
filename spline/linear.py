
import torch

from .binned import BinnedSpline


class LinearSpline(BinnedSpline):
    def __init__(self, *args, bins: int = 10, **kwargs):
        #       parameter                           constraints             count
        # 1.    the domain half-width B             positive                1
        # 2.    the relative width of each bin      positive, sum to 1      #bins
        # 3.    the relative height of each bin     positive, sum to 1      #bins
        super().__init__(*args, **kwargs, bins=bins, parameter_counts=[1, bins, bins])

    def _spline1(self, x: torch.Tensor, *params: torch.Tensor, rev: bool = False) -> torch.Tensor:
        xs, ys = params

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
        xk = torch.gather(xs, dim=-1, index=left).squeeze(-1)
        xkp = torch.gather(xs, dim=-1, index=right).squeeze(-1)
        yk = torch.gather(ys, dim=-1, index=left).squeeze(-1)
        ykp = torch.gather(ys, dim=-1, index=right).squeeze(-1)

        dx = xkp - xk
        dy = ykp - yk

        if not rev:
            xi = (x - xk) / dx
            out = dy / dx * xi + yk
        else:
            y = x

            xi = (y - yk) * dx / dy
            out = xi * dx + xk

        log_jac = torch.log(dy) - torch.log(dx)

        return out, log_jac
