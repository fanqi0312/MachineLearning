import numpy as np

print("================random随机数")
# randn 是从标准正态分布中返回随机数

arr1 = np.random.randn(1)
print(arr1)
# [ 0.21681196]
arr1 = np.random.randn(2)
print(arr1)
# [-0.42666614  0.28317858]
arr1 = np.random.randn(2, 3)
print(arr1)
# [[-1.30036151 -1.07492001  1.02804077]
#  [ 0.91955617 -1.63311815  0.55474165]]
arr1 = np.random.randn(2, 3, 4)
print(arr1)
# [[[-0.44024833 -0.04528447  0.75236555 -0.44405048]
#   [-1.2795628   0.85774872 -0.64444633 -0.22383223]
#   [ 0.80691971  0.77668038  0.56059061 -0.6840716 ]]
#
#  [[-1.46496597  1.49347242 -0.11687603 -0.48794968]
#   [-0.249865   -0.30901606  0.97586369  0.84691122]
#   [-0.0315555   2.61086409 -0.21893868  0.1313964 ]]]


# rand 的随机样本位于（0, 1）之间
arr1 = np.random.rand(1)
print(arr1)
# [ 0.04166528]


# randint 生成的随机数整数n
a = 2
b = 5
# a <= n < b
randintN = np.random.randint(a, b)
print(randintN)

# n < a 的正整数
randintN = np.random.randint(a)
print(randintN)

print("================dot矩阵乘法")
# 维度相同，分别相乘求和
a = [1, 2]
b = [2, 3]
c = [[1, 2], [2, 2]]
npDot = np.dot(a, b)  # 相乘后求和
print(npDot)
npDot = np.dot(b, c)  #
print(npDot)
