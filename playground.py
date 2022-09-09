import matplotlib.pyplot as plt
import numpy as np
import sympy as sym


import torch
from FrEIA.modules.coupling_layers import _BaseCouplingBlock


class CubicSplineCoupling(_BaseCouplingBlock):
    def __init__(self, dims_in, dims_c):
        super().__init__(dims_in, dims_c)
        a = sym.Symbol("a", real=True)
        b = sym.Symbol("b", real=True)
        c = sym.Symbol("c", real=True)
        d = sym.Symbol("d", real=True)
        x = sym.Symbol("x", real=True)
        x0 = sym.Symbol("x_0", real=True)
        x1 = sym.Symbol("x_1", real=True)
        y0 = sym.Symbol("y_0", real=True)
        y1 = sym.Symbol("y_1", real=True)
        d0 = sym.Symbol("d_0", real=True, positive=True)
        d1 = sym.Symbol("d_1", real=True, positive=True)

        f = a * x ** 3 + b * x ** 2 + c * x + d
        df = sym.diff(f, x)

        eq1 = f.subs(x, x0) - y0
        eq2 = f.subs(x, x1) - y1
        eq3 = df.subs(x, x0) - d0
        eq4 = df.subs(x, x1) - d1

        solution = sym.solve([eq1, eq2, eq3, eq4], [a, b, c, d])

    def spline_params(self, lower, upper, derivatives):
        d_0, d_1 = derivatives
        x_0, y_0 = lower
        x_1, y_1 = upper

        # these are the solutions to the cubic equation parameters
        # given the lower and upper point and derivatives
        # solutions are taken straight out of sympy, hence ugly
        a = (d_0 * x_0 - d_0 * x_1 + d_1 * x_0 - d_1 * x_1 - 2 * y_0 + 2 * y_1) / (x_0 ** 3 - 3 * x_0 ** 2 * x_1 + 3 * x_0 * x_1 ** 2 - x_1 ** 3)
        b = (-d_0 * x_0 ** 2 - d_0 * x_0 * x_1 + 2 * d_0 * x_1 ** 2 - 2 * d_1 * x_0 ** 2 + d_1 * x_0 * x_1 + d_1 * x_1 ** 2 + 3 * x_0 * y_0 - 3 * x_0 * y_1 + 3 * x_1 * y_0 - 3 * x_1 * y_1) / (x_0 ** 3 - 3 * x_0 ** 2 * x_1 + 3 * x_0 * x_1 ** 2 - x_1 ** 3)
        c = (2 * d_0 * x_0 ** 2 * x_1 - d_0 * x_0 * x_1 ** 2 - d_0 * x_1 ** 3 + d_1 * x_0 ** 3 + d_1 * x_0 ** 2 * x_1 - 2 * d_1 * x_0 * x_1 ** 2 - 6 * x_0 * x_1 * y_0 + 6 * x_0 * x_1 * y_1) / (x_0 ** 3 - 3 * x_0 ** 2 * x_1 + 3 * x_0 * x_1 ** 2 - x_1 ** 3)
        d = (-d_0 * x_0 ** 2 * x_1 ** 2 + d_0 * x_0 * x_1 ** 3 - d_1 * x_0 ** 3 * x_1 + d_1 * x_0 ** 2 * x_1 ** 2 + x_0 ** 3 * y_1 - 3 * x_0 ** 2 * x_1 * y_1 + 3 * x_0 * x_1 ** 2 * y_0 - x_1 ** 3 * y_0) / (x_0 ** 3 - 3 * x_0 ** 2 * x_1 + 3 * x_0 * x_1 ** 2 - x_1 ** 3)

        return a, b, c, d

    def _coupling(self, x, u, rev=False, *, subnet):
        params = subnet(u)
        lower, upper, derivatives = torch.split(params, [2, 2, 2], dim=1)
        a, b, c, d = self.spline_params(lower, upper, derivatives)

        spline = a * x1 ** 3 + b * x1 ** 2 + c * x1 + d
        lower_tail = derivatives[:, 0] * x + lower[:, 1] - derivatives[:, 0] * lower[:, 0]
        upper_tail = derivatives[:, 1] * x + upper[:, 1] - derivatives[:, 1] * upper[:, 0]

        y1 = spline
        y1 = torch.where(lower[:, 0] <= x, y1, lower_tail)
        y1 = torch.where(x <= upper[:, 0], y1, upper_tail)

        return y1

    def _coupling2(self, x2, u1, rev=False):
        params = self.subnet2(u1)
        lower, upper, derivativess = torch.split(params, [2, 2, 2], dim=1)





def spline(x, lower, upper, derivatives):
    # x: (n,)
    # lower: (2,)
    # upper: (2,)
    # derivatives: (2,)

    lower_tail = derivatives[0] * x + lower[1] - derivatives[0] * lower[0]
    upper_tail = derivatives[1] * x + upper[1] - derivatives[1] * upper[0]

    d_0, d_1 = derivatives
    x_0, y_0 = lower
    x_1, y_1 = upper

    # these monstrosities come straight out of sympy
    a = (d_0 * x_0 - d_0 * x_1 + d_1 * x_0 - d_1 * x_1 - 2 * y_0 + 2 * y_1) / (x_0 ** 3 - 3 * x_0 ** 2 * x_1 + 3 * x_0 * x_1 ** 2 - x_1 ** 3)
    b = (-d_0 * x_0 ** 2 - d_0 * x_0 * x_1 + 2 * d_0 * x_1 ** 2 - 2 * d_1 * x_0 ** 2 + d_1 * x_0 * x_1 + d_1 * x_1 ** 2 + 3 * x_0 * y_0 - 3 * x_0 * y_1 + 3 * x_1 * y_0 - 3 * x_1 * y_1) / (x_0 ** 3 - 3 * x_0 ** 2 * x_1 + 3 * x_0 * x_1 ** 2 - x_1 ** 3)
    c = (2 * d_0 * x_0 ** 2 * x_1 - d_0 * x_0 * x_1 ** 2 - d_0 * x_1 ** 3 + d_1 * x_0 ** 3 + d_1 * x_0 ** 2 * x_1 - 2 * d_1 * x_0 * x_1 ** 2 - 6 * x_0 * x_1 * y_0 + 6 * x_0 * x_1 * y_1) / (x_0 ** 3 - 3 * x_0 ** 2 * x_1 + 3 * x_0 * x_1 ** 2 - x_1 ** 3)
    d = (-d_0 * x_0 ** 2 * x_1 ** 2 + d_0 * x_0 * x_1 ** 3 - d_1 * x_0 ** 3 * x_1 + d_1 * x_0 ** 2 * x_1 ** 2 + x_0 ** 3 * y_1 - 3 * x_0 ** 2 * x_1 * y_1 + 3 * x_0 * x_1 ** 2 * y_0 - x_1 ** 3 * y_0) / (x_0 ** 3 - 3 * x_0 ** 2 * x_1 + 3 * x_0 * x_1 ** 2 - x_1 ** 3)

    spline = a * x ** 3 + b * x ** 2 + c * x + d

    print(a, b, c, d)

    y = spline
    y = np.where(lower[0] <= x, y, lower_tail)
    y = np.where(x <= upper[0], y, upper_tail)

    return y


x = np.linspace(-10, 10, 1000)
lower = np.array([0, 1])
upper = np.array([5, -2])
widths = np.array([0.25, 0.45, 0.3])
heights = np.array([0.5, 0.1, 0.4])
derivatives = np.array([0.1, 2.0])

y = spline(x, lower, upper, derivatives)
invertible = np.allclose(y, np.sort(y))
print("Invertible?", invertible)

plt.plot(x, y)
rect = plt.Rectangle(xy=lower, width=upper[0] - lower[0], height=upper[1] - lower[1], lw=1, color="black", fill=False)
plt.gca().add_patch(rect)
plt.show()

a = sym.Symbol("a", real=True)
b = sym.Symbol("b", real=True)
c = sym.Symbol("c", real=True)
d = sym.Symbol("d", real=True)
x = sym.Symbol("x", real=True)
x0 = sym.Symbol("x_0", real=True)
x1 = sym.Symbol("x_1", real=True)
y0 = sym.Symbol("y_0", real=True)
y1 = sym.Symbol("y_1", real=True)
d0 = sym.Symbol("d_0", real=True, positive=True)
d1 = sym.Symbol("d_1", real=True, positive=True)

f = a * x ** 3 + b * x ** 2 + c * x + d

# df = sym.diff(f, x)
#
# eq1 = f.subs(x, x0) - y0
# eq2 = f.subs(x, x1) - y1
# eq3 = df.subs(x, x0) - d0
# eq4 = df.subs(x, x1) - d1
#
# solution = sym.solve([eq1, eq2, eq3, eq4], [a, b, c, d])
#
# print(solution[a])
#
# get_a = sym.lambdify((x0, x1, y0, y1, d0, d1), solution[a], "numpy")
#
# print(help(get_a))
#
# for key, sol in solution.items():
#     print(type(key))
#     print(f"{key} = {sym.collect(sol, [x0, x1])}")


# rational quadratic parameters
a1, a2, b1, b2, c1, c2 = sym.symbols("a1, a2, b1, b2, c1, c2", real=True)

print(:-1)
