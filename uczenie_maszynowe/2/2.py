import numpy as np
import matplotlib.pyplot as plt
import sys

# 873
epsilon = sys.float_info.epsilon

plt.xlabel('x')
plt.xlabel('y')

x1 = np.arange(-5.0, 5.0, 0.25)
y1 = 4 * x1**2 + 2 * x1 - 3

x2 = np.arange(-5.0, 5.0, 0.25)
y2 = epsilon**x2 / (epsilon**x2 + 1)

plt.plot(x1, y1, x2, y2)
plt.show()