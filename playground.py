
from FrEIA.modules import LinearSpline, RationalQuadraticSpline

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

torch.autograd.set_grad_enabled(False)

batch_size = 1000
input_shape = (2,)
bins = 4

B = 3
xs = torch.Tensor([-B, -2, 0, 1, B])
ys = torch.Tensor([-B, -0.5, 1, 2, B])
deltas = torch.Tensor([4, 0.2, 3])

def subnet_constructor(in_features, out_features):

    b = np.log(np.exp(B) - 1) - np.log(np.e - 1)
    d = np.log(np.exp(deltas) - 1) - np.log(np.e - 1)

    widths = torch.log(xs[1:] - xs[:-1])
    heights = torch.log(ys[1:] - ys[:-1])

    params = torch.Tensor([
        b,
        *widths,
        *heights,
        *d
    ])

    def subnet(x):
        p = params.repeat(x.shape[0], 1)
        return p

    return subnet


x = torch.linspace(-(B + 1), B + 1, steps=batch_size)

x = x[[None] * len(input_shape)].movedim(-1, 0)

x = x.expand(batch_size, *input_shape).contiguous()
x = (x,)

# rqs = LinearSpline(dims_in=(input_shape,), subnet_constructor=subnet_constructor, bins=bins)
rqs = RationalQuadraticSpline(dims_in=(input_shape,), subnet_constructor=subnet_constructor, bins=bins)


z, _ = rqs.forward(x)

x = x[0]
z = z[0]

print(f"{z.shape=}")

plt.figure(dpi=250)
rect = plt.Rectangle(xy=(-B, -B), width=2 * B, height=2 * B, color="black", fill=False, alpha=0.2, ls="--")
plt.gca().add_patch(rect)

plt.plot(x[:, 0], x[:, 0], ls="--", color="black", alpha=0.2)
plt.plot(x[:, 0], z[:, 0], label="Forward")
plt.plot(z[:, 0], x[:, 0], label="Inverse")
plt.scatter(xs, ys, color="black", marker="o")
plt.scatter(ys, xs, color="black", marker="o")

plt.legend()
plt.xlabel("Input")
plt.ylabel("Output")
plt.tight_layout()
plt.savefig("rq_spline.png")
plt.show()

