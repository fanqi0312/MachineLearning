"""
Python 内置函数

"""

print("================================len长度")
len([1, 2])  # 2
len([4, 5, 6])  # 3
[1, 2].__len__()  # 2


print("================================zip")
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
xyz = zip(x, y, z)  # 注意：xyz是对象

print(list(xyz))
# [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

x = [1, 2]  # 返回长度最短，6,9忽略
y = [4, 5, 6]
z = [7, 8, 9]
xyz = zip(x, y, z)

print(list(xyz))
# [(1, 4, 7), (2, 5, 8)]


print("================================range序列")
range(5)
# 代表从0到5(不包含5),[0, 1, 2, 3, 4]
range(1, 5)
# 代表从1到5(不包含5),[1, 2, 3, 4]
range(1, 5, 2)
# 代表从1到5，间隔2(不包含5),[1, 3]

# 应用场景
for i in range(1, 4):
    print(i)
    #1,2,3


print("================================range序列")
abs(-10)
# 10




# from sympy.printing.tests.test_numpy import np
# layers = [2,2,1]
# weights = []
# for i in range(1, len(layers) - 1):
#     weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
#     print(weights)
#     weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)
#     print(weights)