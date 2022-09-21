
import torch

from FrEIA.modules.coupling_layers import _BaseCouplingBlock

import utils


from .base import SplineCoupling


class TailedSpline(SplineCoupling):
    def __init__(self, dims_in, dims_c=None, inner: SplineCoupling = None):
        if dims_c is None:
            dims_c = []
        super().__init__(dims_in, dims_c)

        self.inner = inner

    def _coupling1(self, x1, u2, rev=False):
        params = self.subnet1(u2)

        domain, other_params = torch.split(params, [...], dim=1)

        inside = ...  # x in domain

        spline_out, spline_log_jac = self.inner._spline1(x1, domain, other_params)

        # affine tails
        a = ...
        b = ...
        out = a * x1 + b
        # overwrite inside with spline
        out[inside] = spline_out
        # same for jacobian
        log_jac = a
        log_jac[inside] = spline_log_jac

        log_jac_det = utils.sum_except_batch(log_jac)

        if rev:
            log_jac_det = -log_jac_det

        return out, log_jac_det

    def _coupling2(self, x2, u1, rev=False):
        pass
