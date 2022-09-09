from typing import List, Tuple

import FrEIA.modules as fm

from .rational_quadratic import RationalQuadraticSpline


class ARQ(fm.InvertibleModule):
    """
    Affine Rational-Quadratic Coupling Block
    This is simply an affine coupling followed by a rational-quadratic spline coupling
    """

    def __init__(self, dims_in, dims_c=None, affine_subnet_constructor=None, spline_subnet_constructor=None, bins=10):
        if dims_c is None:
            dims_c = []
        super().__init__(dims_in, dims_c)

        self.affine = fm.GLOWCouplingBlock(dims_in, dims_c=dims_c, subnet_constructor=affine_subnet_constructor)
        self.spline = RationalQuadraticSpline(dims_in, dims_c=dims_c, subnet_constructor=spline_subnet_constructor, bins=bins)

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        return input_dims

    def forward(self, x_or_z, c=None, rev=False, jac=True):
        x_or_z, log_jac_det1 = self.affine.forward(x_or_z, c, rev, jac)
        x_or_z, log_jac_det2 = self.spline.forward(x_or_z, c, rev, jac)

        return x_or_z, log_jac_det1 + log_jac_det2
