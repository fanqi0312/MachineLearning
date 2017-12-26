

# ================zip

x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
xyz = zip(x, y, z) # 注意：xyz是对象

print(list(xyz))
# [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

x = [1, 2 ]  #返回长度最短，6,9忽略
y = [4, 5, 6]
z = [7, 8, 9]
xyz = zip(x, y, z)

print(list(xyz))
# [(1, 4, 7), (2, 5, 8)]

