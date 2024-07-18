
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns


import numpy as np
from scipy.linalg import hadamard, subspace_angles
rng = np.random.default_rng()
H = hadamard(4)

A = H[:, 2:]
B = H[:, :2]
print('A',A.shape)
print('B',B.shape)

angle = np.rad2deg(subspace_angles(A, B))
print(angle)



