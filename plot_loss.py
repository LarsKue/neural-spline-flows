
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


def tensorboard_smooth(x: pd.Series, strength: float = 0.6):
    return x.ewm(alpha=1 - strength).mean()

dataset = "circles"

affine = pd.read_csv(f"loss/{dataset}_affine.csv")
spline = pd.read_csv(f"loss/{dataset}_rq_spline.csv")

smoothing_strength = 0.7

x = affine["Step"]
y = affine["Value"]
smooth_y = tensorboard_smooth(y, strength=smoothing_strength)

ax = sns.lineplot(x=x, y=y, alpha=0.25, color="C0")
sns.lineplot(x=x, y=smooth_y, color="C0", label="Affine")

x = spline["Step"]
y = spline["Value"]
smooth_y = tensorboard_smooth(y, strength=smoothing_strength)

sns.lineplot(x=x, y=y, ax=ax, alpha=0.25, color="C1")
sns.lineplot(x=x, y=smooth_y, ax=ax, color="C1", label="RQ Spline")

plt.title(dataset.capitalize())
ax.set_xlabel("Step")
ax.set_ylabel("Smoothed Validation Loss")
# plt.ylim(0.3, 0.6)  # moons
plt.ylim(0.6, 1.0)  # circles
plt.savefig(f"plots/loss_{dataset}.png")
plt.show()
