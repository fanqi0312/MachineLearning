"""
Numpy 库用法

"""

import numpy as np


print("================================shape矩阵维度")

c = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(c.shape)
# (4, 2)
print(c.shape[0])
# 4
print(c.shape[1])
# 2
