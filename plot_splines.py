
from FrEIA.modules import LinearSpline, RationalQuadraticSpline

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

torch.autograd.set_grad_enabled(False)

batch_size = 1000
input_shape = (2,)
bins = 4

left = -3
bottom = -5
width = 5
height = 8

right = left + width
top = bottom + height

xs = torch.Tensor([left, -1, 0, 1, right])
ys = torch.Tensor([bottom, 0, 0.5, 1.5, top])
deltas = torch.Tensor([3, 0.25, 2])

min_bin_sizes = (0.0, 0.0)
default_domain = (-1, 1, -1, 1)

def subnet_constructor(in_features, out_features):

    l = left + 1
    b = bottom + 1

    default_width = 2
    default_height = 2

    xshift = np.log(np.exp(default_width) - 1)
    yshift = np.log(np.exp(default_height) - 1)

    bin_widths = xs[1:] - xs[:-1]
    bin_heights = ys[1:] - ys[:-1]
    print(f"{bin_widths=}")
    print(f"{bin_heights=}")

    bin_widths = np.log(np.exp(bin_widths) - 1) - xshift
    bin_heights = np.log(np.exp(bin_heights) - 1) - yshift

    # d = np.zeros_like(deltas)
    d = np.log(np.exp(deltas) - 1)

    params = torch.Tensor([
        l,
        b,
        *bin_widths,
        *bin_heights,
        *d,
    ])

    # params = torch.Tensor([
    #     l, b, w, h,
    #     *bin_widths,
    #     *bin_heights,
    #     *d
    # ])

    def subnet(x):
        # return x.new_zeros(x.shape[0], params.shape[0])
        p = params.repeat(x.shape[0], 1)
        return p

    return subnet


x = torch.linspace(left - 1, right + 1, steps=batch_size)

x = x[[None] * len(input_shape)].movedim(-1, 0)

x = x.expand(batch_size, *input_shape).contiguous()
x = (x,)

# spline = LinearSpline(dims_in=(input_shape,), subnet_constructor=subnet_constructor, bins=bins, default_domain=default_domain, min_bin_sizes=(0.0, 0.0))
spline = RationalQuadraticSpline(dims_in=(input_shape,), subnet_constructor=subnet_constructor, bins=bins, default_domain=default_domain, min_bin_sizes=min_bin_sizes)


z, logdet = spline.forward(x)
zz, _ = spline.forward(x, rev=True)
xx, _ = spline.forward(z, rev=True)

x = x[0]
xx = xx[0]
z = z[0]
zz = zz[0]

plt.figure(dpi=250)
rect = plt.Rectangle(xy=(left, bottom), width=width, height=height, color="black", fill=False, alpha=0.2, ls="--")
plt.gca().add_patch(rect)

plt.plot(x[:, 0], x[:, 0], ls="--", color="black", alpha=0.2)
plt.plot(x[:, 0], z[:, 0], label="Forward")

plt.plot(z[:, 0], x[:, 0], label="Inverse")
# plt.plot(x[:, 0], zz[:, 0], label="Actual Inverse")
# plt.plot(x[:, 0], xx[:, 0], label="Should be Diagonal")
plt.scatter(xs, ys, color="black", marker="o")
plt.scatter(ys, xs, color="black", marker="o")

plt.legend()
plt.title("Rational-Quadratic Spline")
plt.xlabel("Input")
plt.ylabel("Output")

# ax2 = plt.gca().twinx()
# ax2.plot(x[:, 0], np.exp(logdet), color="black", alpha=0.2)
# ax2.plot(x[:, 0], np.ones_like(x[:, 0]), color="black", alpha=0.2, ls=":")
# ax2.set_ylabel("Jacobian")

plt.tight_layout()
plt.savefig("plots/rq_spline.png")
plt.show()




