
import torch
from typing import Callable


def tailed_spline(x: torch.Tensor, *, params: torch.Tensor, bins: int, spline: Callable, rev: bool = False):
    """
    Apply the given spline inside the spline domain, with identity tails outside the spline domain
    :param x: input tensor
    :param params: unconstrained spline parameters
    :param bins: number of bins to use
    :param spline: spline function
    :param rev: whether to run in reverse
    :return: tuple containing the output tensor and the log jacobian determinant
    """
    xs, ys, deltas = transform_spline_params(params, bins=bins)

    # bin edge cases are the same as in torch.searchsorted
    inside = (xs[..., 0] < x) & (x <= xs[..., -1])

    # pass inputs inside spline boundary to spline
    spline_out, spline_log_jac = spline(x[inside], xs[inside], ys[inside], deltas[inside], rev=rev)

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

from .binned import BinnedSpline


class TailedSpline(BinnedSpline):
    def _spline1(self, x: torch.Tensor, x_left: torch.Tensor, x_right: torch.Tensor, y_left: torch.Tensor, y_right: torch.Tensor, *params: torch.Tensor, rev: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys, *_ = params

        # TODO: make this a wrapper or something maybe

        # bin edge cases are the same as in torch.searchsorted
        inside = (xs[..., 0] < x) & (x <= xs[..., -1])

        # pass inputs inside spline boundary to spline
        spline_out, spline_log_jac = spline(x[inside], xs[inside], ys[inside], deltas[inside], rev=rev)

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


