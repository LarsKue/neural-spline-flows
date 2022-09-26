
import torch
import torch.nn.functional as F

import numpy as np

from .binned import BinnedSpline


class RationalQuadraticSpline(BinnedSpline):
    def __init__(self, *args, bins: int = 10, **kwargs):
        #       parameter                                       constraints             count
        # 1.    the domain half-width B                         positive                1
        # 2.    the relative width of each bin                  positive, sum to 1      #bins
        # 3.    the relative height of each bin                 positive, sum to 1      #bins
        # 4.    the derivative at the edge of each inner bin    positive                #bins - 1
        super().__init__(*args, **kwargs, bins=bins, parameter_counts=[1, bins, bins, bins - 1])

    def _constrain_params1(self, *params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        domain_width, bin_widths, bin_heights, deltas, *params = super()._constrain_params1(*params)

        shift = np.log(np.e - 1)
        deltas = F.softplus(deltas + shift)

        return domain_width, bin_widths, bin_heights, deltas, *params

    def _constrain_params2(self, *params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        domain_width, bin_widths, bin_heights, deltas, *params = super()._constrain_params2(*params)

        shift = np.log(np.e - 1)
        deltas = F.softplus(deltas + shift)

        return domain_width, bin_widths, bin_heights, deltas, *params

    def _spline1(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, bottom: torch.Tensor, top: torch.Tensor, *params: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

        # TODO:
        delta_left = ...
        delta_right = ...

        return rational_quadratic_spline(x, left, right, bottom, top, delta_left, delta_right, rev=rev)

    def _spline2(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, bottom: torch.Tensor, top: torch.Tensor, *params: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        delta_left, delta_right = params
        return rational_quadratic_spline(x, left, right, bottom, top, delta_left, delta_right, rev=rev)


def rational_quadratic_spline(x: torch.Tensor,
                              left: torch.Tensor,
                              right: torch.Tensor,
                              bottom: torch.Tensor,
                              top: torch.Tensor,
                              delta_left: torch.Tensor,
                              delta_right: torch.Tensor,
                              rev: bool = False,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rational-quadratic spline with the algorithm described in arXiv:1906.04032
    Forward output is defined for each bin by the fraction
    ..math::
        z = f(x) = \\frac{ \\beta_0 + \\beta_1 x + \\beta_2 x^2 }{ 1 + \\beta_3 x + \\beta_4 x^2 }

    where the \\beta are constrained to yield a smooth function over all bins.

    :param x: input tensor of shape (batch, ...)
    :param left: bin edges to the left of x for each input, shape (batch, ..., 1)
    :param right: bin edges to the right of x for each input, shape (batch, ..., 1)
    :param bottom: bin edges below x for each input, shape (batch, ..., 1)
    :param top: bin edges above x for each input, shape (batch, ..., 1)
    :param delta_left: spline derivative at the left bin edge of shape (batch, ..., 1)
    :param delta_right: spline derivative at the right bin edge of shape (batch, ..., 1)
    :param rev: whether to run in reverse
    :return: tuple containing the output tensor and the log jacobian determinant
    """

    # rename variables to match the paper:
    # xk means $x_k$ and xkp means $x_{k+1}$
    xk = left
    xkp = right
    yk = bottom
    ykp = top
    dk = delta_left
    dkp = delta_right

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
