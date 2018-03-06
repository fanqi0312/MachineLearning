"""
Numpy 库用法

"""

import numpy as np

print("================================ 技巧")
# 打印帮助文档，也可直接点击
# Ctrl右侧显示代码
# print(help(np.genfromtxt))

print("================================ genfromtxt打开数据")
# 参数:
# 文件路径:
# delimiter:分割符号
# dtype:文件格式,也可"U75"
# skip_header:跳过行数据，如跳过标题行
world_alcohol = np.genfromtxt("data/world_alcohol.txt", delimiter=",", dtype=str, skip_header=1)
print(world_alcohol)
# [['Year' 'WHO region' 'Country' 'Beverage Types' 'Display Value']
#  ['1986' 'Western Pacific' 'Viet Nam' 'Wine' '0']
#  ['1986' 'Americas' 'Uruguay' 'Other' '0.5']
print(type(world_alcohol))  # <class 'np.ndarray'>

# 取数据
uruguay_other_1986 = world_alcohol[1, 4]
print(uruguay_other_1986)  # 获取第2行，第5列的0.5

third_country = world_alcohol[2, 2]
print(third_country)  # 获取第3行，第3列的Cte d'Ivoire

print("================================ array数组")
# The np.array() function can take a list or list of lists as input. When we input a list, we get a one-dimensional array as a result:
vector = np.array([5, 10, 15, 20])
print(vector)

# When we input a list of lists, we get a matrix as a result:
matrix = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
print(matrix)

print("================================ shape矩阵维度")
c = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(c.shape)
# (4, 2)
print(c.shape[0])
# 4
print(c.shape[1])
# 2

# We can use the ndarray.shape property to figure out how many elements are in the array
vector = np.array([1, 2, 3, 4])
print(vector.shape)  # (4,)
# For matrices, the shape property contains a tuple with 2 elements.
matrix = np.array([[5, 10, 15], [20, 25, 30]])
print(matrix.shape)  # (2, 3)

print("================================ array数组")
# 数据格式必须相同
# Each value in a NumPy array has to have the same data type
# NumPy will automatically figure out an appropriate data type when reading in data or converting lists to arrays.
# You can check the data type of a NumPy array using the dtype property.
numbers = np.array([1, 2, 3, 4])
print(numbers.dtype)  # int32
print(numbers)  # [1 2 3 4]

numbers = np.array([1, 2, 3, 4.0])
print(numbers.dtype)  # float64
print(numbers)  # [ 1.  2.  3.  4.]

numbers = np.array([1, 2, 3, "4"])
print(numbers.dtype)  # <U11
print(numbers)  # ['1' '2' '3' '4']

print("================================ 切片")
vector = np.array([5, 10, 15, 20])
print(vector[0:3])  # 前3个（不含第4个），[ 5 10 15]

matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
print(matrix[:, 1])  # 1列，[10 25 40]

matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
print(matrix[:, 0:2])  # 0、1列
# [[ 5 10]
#  [20 25]
#  [35 40]]

matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
print(matrix[1:3, 0:2])  # 1、2行，0、1列
# [[20 25]
#  [35 40]]


print("================================ 数组判断和取值")
# it will compare the second value to each element in the vector
# If the values are equal, the Python interpreter returns True; otherwise, it returns False
# 对数组每个元素判断
vector = np.array([5, 10, 15, 20])
equal_to_ten = (vector == 10)
print(equal_to_ten)  # [False  True False False]
# 作为索引取出10
# Compares vector to the value 10, which generates a new Boolean vector [False, True, False, False]. It assigns this result to equal_to_ten
print(vector[equal_to_ten])  # [10]

matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
print(matrix == 25)
# [[False False False]
#  [False  True False]
#  [False False False]]

second_column_25 = (matrix[:, 1] == 25)  # 1列==25的
print(second_column_25)  # [False  True False]

# 取出所有行数据
print(matrix[second_column_25, :])  # [[20 25 30]]

print("================================ 数组与或判断")

# We can also perform comparisons with multiple conditions
vector = np.array([5, 10, 15, 20])
# 某个元素即等于10又等于5（不可能）
equal_to_ten_and_five = (vector == 10) & (vector == 5)
print(equal_to_ten_and_five)  # [False False False False]

equal_to_ten_or_five = (vector == 10) | (vector == 5)
print(equal_to_ten_or_five)  # [ True  True False False]

equal_to_ten_or_five = (vector == 10) | (vector == 5)
# 指定数值重新赋值
vector[equal_to_ten_or_five] = 50
print(vector)  # [50 50 15 20]

matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
# 取出25所在的列
second_column_25 = matrix[:, 1] == 25
print(second_column_25)  # [False  True False]
matrix[second_column_25, 1] = 10
print(matrix)
# [[ 5 10 15]
#  [20 10 30]
#  [35 40 45]]


print("================================ 类型转换")
# We can convert the data type of an array with the ndarray.astype() method.
vector = np.array(["1", "2", "3"])
print(vector.dtype)  # <U1
print(vector)  # ['1' '2' '3']
# 转换为Float类型
vector = vector.astype(float)
print(vector.dtype)  # float64
print(vector)  # [ 1.  2.  3.]

print("================================ 极值")
vector = np.array([5, 10, 15, 20])
print(vector.min())  # 最小值 5
print(vector.max())  # 最大值 20

print("================================ 求和")
vector = np.array([5, 10, 15, 20])
print(vector.sum())  # 50

# The axis dictates which dimension we perform the operation on
# 1 means that we want to perform the operation on each row, and 0 means on each column
matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45],
    [50, 55, 60]
])
# 按列求和
print(matrix.sum(axis=0))  # [110 130 150]
# 按行求和
print(matrix.sum(axis=1))  # [ 30  75 120 165]


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

print("================================ 构造矩阵")
#值为0的矩阵
np.zeros((3, 4))

#值为1的矩阵
np.ones((2, 3, 4), dtype=np.int32)

#随机矩阵
a = np.random.random((2, 3))
print(a)
# [[0.97994275 0.54939347 0.67802913]
#  [0.85679225 0.1220005  0.84893534]]

#序列矩阵，从0-14的数组，加步长
a = np.arange(15)
print(a)  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

# 从10-30的每5生成
a = np.arange(10, 30, step=5)  # [10 15 20 25]
a = np.arange(10, 30, 5)  # [10 15 20 25]
# print(a)

a = np.arange(0, 2, 0.3)  # [ 0.   0.3  0.6  0.9  1.2  1.5  1.8]
# print(a)


from numpy import pi

# 区间平均生成数量，100个
a = np.linspace(0, 2 * pi, 100)
# print(a)
# 科学计数
a = np.sin(np.linspace(0, 2 * pi, 100))
# print(a)

a = np.floor(10 * np.random.random((2, 2)))
b = np.floor(10 * np.random.random((2, 2)))




print("================================ 排列矩阵")
a = np.arange(15) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

# reshape
# 分割为3行5列矩阵
a = a.reshape(3, 5)
# -1为自动计算
# a = a.reshape(3, -1)
print(a)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]

print(a.shape)  # (3, 5)

print(a.ndim)  # 2

print(a.dtype.name)  # int32

print(a.size)  # 15

# 洗牌（只洗行，变换数据索引）
np.random.shuffle(a)
print(a)
# [[10 11 12 13 14]
#  [ 0  1  2  3  4]
#  [ 5  6  7  8  9]]

# T行列替换
print(a.T)
# [[ 0  5 10]
#  [ 1  6 11]
#  [ 2  7 12]
#  [ 3  8 13]
#  [ 4  9 14]]



# 矩阵还原为数组
print(a.ravel())  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]



print("================================ 计算")
# the product operator * operates elementwise in NumPy arrays
a = np.array([20, 30, 40, 50])  # [20 30 40 50]
b = np.arange(4)  # [0 1 2 3]
print(a)
print(b)
# 对应位置相减（同纬度）
c = a - b  # [20 29 38 47]
print(c)

# 每个数都减1
c = a - 1  # [19 29 39 49]
print(c)
# 每个数都乘2
c = b ** 2  # [0 1 4 9]
print(c)
# 每个数都判断
print(a < 35)  # [ True  True False False]



# The matrix product can be performed using the dot function or method
A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])
C = np.array([2, 3])

# 矩阵点乘（对应位置相乘）
print(A * B)
# [[2 0]
#  [0 4]]


# 维度相同，矩阵乘法，行乘列
print(A.dot(B))
# [[5 4]
#  [3 4]]

print(B.dot(A))
# [[2 2]
#  [3 7]]

# 维度不同
print(np.dot(B,C))  # np.dot()运算效果相同
#[ 4 18]

print(np.dot(C,B))
# [13 12]

print("================================ 运算符")
B = np.array([0, 1, 2])
# e为底的指数函数
print(np.exp(B))  # [ 1. 2.71828183 7.3890561 ]
# 根号 平方根
print(np.sqrt(B))  # [ 0. 1. 1.41421356]

a = np.array([[0.12, 1.12, 2.56],
              [10.12, 12.12, 15.56]])
# 向下取整
# Return the floor of the input
a = np.floor(a)
print(a)


# 重新定义维度
try:
    print(a.resize(( 6,2)))
except:
    print('resize错误')
# [[  0.   1.   2.  10.  12.  15.]
#  [  0.   0.   0.   0.   0.   0.]]


print("================================ 拼接与分割")
a = np.array([[1, 2],
              [5, 6]])
b = np.array([[3, 4],
              [7, 8]])
# 行向拼接
c = np.hstack((a, b))
print(c)
# [[1 2 3 4]
#  [5 6 7 8]]

# 列向拼接
print(np.vstack((a, b)))
# [[1 2]
#  [5 6]
#  [3 4]
#  [7 8]]

# 列切分
print(np.hsplit(c, 2))
# [array([[1, 2],[5, 6]]),
#  array([[3, 4],[7, 8]])]

print(np.hsplit(c, (2, 2)))
# [array([[1, 2],[5, 6]]),
#  array([], shape=(2, 0), dtype=int32),
#  array([[2],[6]])]

# 行切分
print(np.vsplit(c, 2))
# [array([[1, 2, 3, 4]]),
#  array([[5, 6, 7, 8]])]


print("================================ 复制")
# Simple assignments make no copy of array objects or of their data.
a = np.arange(12)
# 1. 地址复制。ab指向相同地址，会同时修改
b = a
# a and b are two names for the same ndarray object
b is a
b.shape = 3, 4
print(a.shape)
print(id(a))
print(id(b))

# 2. 值地址复制。数组中的同一个值会同时
# The view method creates a new array object that looks at the same data.
c = a.view()
c is a
c.shape = 2, 6
# print(a.shape
c[0, 4] = 1234
a

# 3. 完全复制，没有任何关系（常用）
# The copy method makes a complete copy of the array and its data.
d = a.copy()
d is a
d[0, 0] = 9999
print(d)
print(a)

print("================================ argmax最大值，tile行列复制")
data = np.sin(np.arange(20)).reshape(5, 4)
print(data)
# 获取最大值的索引
ind = data.argmax(axis=0)  # [2 0 3 1]
print(ind)

data_max = data[ind, range(data.shape[1])]
print(data_max)
all(data_max == data.max(axis=0))

a = np.arange(0, 40, 10)
# 行列复制
b = np.tile(a, (3, 5))
print(b)

a = np.array([[4, 3, 5], [1, 2, 1]])
# 排序
b = np.sort(a, axis=1)
print(b)

a.sort(axis=1)
# print(a)
a = np.array([4, 3, 1, 2])
j = np.argsort(a)
print(j)
print(a[j])

# If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated:
# a.reshape(3,-1)


# When NumPy can't convert a value to a numeric data type like float or integer, it uses a special nan value that stands for Not a Number
# nan is the missing data
# 1.98600000e+03 is actually 1.986 * 10 ^ 3
world_alcohol

print("================================ 过期")

# replace nan value with 0
world_alcohol = np.genfromtxt("data/world_alcohol.txt", delimiter=",")
# print(world_alcohol
is_value_empty = np.isnan(world_alcohol[:, 4])
# print(is_value_empty
world_alcohol[is_value_empty, 4] = '0'
alcohol_consumption = world_alcohol[:, 4]
alcohol_consumption = alcohol_consumption.astype(float)
total_alcohol = alcohol_consumption.sum()
average_alcohol = alcohol_consumption.mean()
print(total_alcohol)
print(average_alcohol)
