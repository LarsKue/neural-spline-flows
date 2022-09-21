
from FrEIA.modules.coupling_layers import _BaseCouplingBlock


class SplineCoupling(_BaseCouplingBlock):

    def _spline1(self, *args, **kwargs):
        raise NotImplementedError

    def _spline2(self, *args, **kwargs):
        raise NotImplementedError

    def _coupling1(self, x1, u2, rev=False):
        params = self.subnet1(u2)

        return self._spline1(...)

    def _coupling2(self, x2, u1, rev=False):
        params = self.subnet2(u1)

        return self._spline2(...)
