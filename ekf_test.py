import numpy as np

from sebekf import ekf

np.diag([3]*3)

pure = np.linspace(-1, 1, 100)
noise = np.random.normal(0, 1, 100)
signal = pure + noise
