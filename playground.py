
import numpy as np
import matplotlib.pyplot as plt

size = 100
x = np.linspace(-3, 3, size)
y = np.random.normal(0, 3, size)
y = np.sort(y)

plt.plot(x, y)
plt.plot(y, x)
plt.show()


