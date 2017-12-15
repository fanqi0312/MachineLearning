print('hello, world')
# hello, world

print(100 + 200)
# 300

print('100 + 200 =', 100 + 200)
# 100 + 200 = 300

# 用逗号“,”隔开，会依次打印每个字符串，遇到逗号“,”会输出一个空格
print('The quick brown fox', 'jumps over', 'the lazy dog')
# The quick brown fox jumps over the lazy dog


# input(),可以让用户输入字符串，并存放到一个变量里。
# name = input('please enter your name: ')
# print('hello,', name)
# please enter your name: fanqi
# hello, fanqi

# :结尾时，缩进的语句视为代码块。
a = 100
if a >= 0:
    print(a)
else:
    print(-a)

# 整数
# 按照科学记数法:1.23x109就是1.23e9
a = 1.23e9

# 字符串
# 以单引号'或双引号",可以用转义字符\来标识
print('I\'m \"OK\"!')
# I'm "OK"!

# 布尔值
# 只有True、False两种值,注意大小写
print(3 > 2)
# True

# and、or和not运算
print(True and False)
# False
print(True or False)
# True

# not单目运算符
print(not True)
# False


### list集合 ###
classmates = ['Michael', 'Bob', 'Tracy']
print(['Michael', 'Bob', 'Tracy'])
# ['Michael', 'Bob', 'Tracy']

print(len(classmates), )
# 3

# 第1个
classmates[0]
# Michael

# 倒数第1个
classmates[-1]
# Tracy

### 切片
# 创建一个0-99的数列
L = list(range(100))
# [0, 1, 2, 3, ..., 99]
# 前2个元素,L[0:2]表示，从索引0开始取，直到索引2为止，但不包括索引2。即索引0，1，正好是2个元素。
L[0:2]
# ['0', '1']
# 如果第一个索引是0，还可以省略：
L[:2]

# 倒数2个
L[-2:]
# [98, 99]

# 前11-20个数
L[10:20]
# [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# 前10个数，每两个取一个
L[:10:2]
# [0, 2, 4, 6, 8]


# 追加元素到末尾
classmates.append('Adam')
# ['Michael', 'Bob', 'Tracy', 'Adam']

# 把元素插入到指定的位置
classmates.insert(1, 'Jack')
# ['Michael', 'Jack', 'Bob', 'Tracy', 'Adam']

# 删除末尾的元素
classmates.pop()
# ['Michael', 'Jack', 'Bob', 'Tracy']

# 删除指定位置的元素
classmates.pop(1)
# ['Michael', 'Bob', 'Tracy']

# 要把某个索引位置元素替换成别的元素
classmates[1] = 'Sarah'
# ['Michael', 'Sarah', 'Tracy']


# list里面的元素的数据类型也可以不同
L = ['Apple', 123, True]

# list元素也可以是另一个list
s = ['python', 'java', ['asp', 'php'], 'scheme']
len(s)  # 4
s[2][1]  # php

### tuple ###
# tuple和list非常类似，但是tuple一旦初始化就不能修改。没有append()，insert()这样的方法
classmates = ('Michael', 'Bob', 'Tracy')
# 一个元素时，要加,号。避免(1)的歧义。
t = (1,)

### 条件判断 ###
# Python的缩进规则，如果if语句判断是True，就把缩进的两行print语句执行了，否则，什么也不做。
age = 20
if age >= 18:
    print('your age is', age)
    print('adult')
elif age >= 6:
    print('teenager')
else:
    print('your age is', age)
    print('teenager')

### 循环 ###
names = ['Michael', 'Bob', 'Tracy']
for name in names:
    print(name)

d = {'a': 1, 'b': 2, 'c': 3}
for key in d:
    print(key,d[key])



# dict 语言中也称为map，使用键-值（key-value）存储
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
d['Michael']
# 95

# 更新值
d['Michael'] = 99

# set 是一组key的集合，但不存储value。由于key不能重复，所以，在set中，没有重复的key。
s = set([1, 2, 3])
# 删除指定元素
s.remove(2)



### 函数
# 绝对值
abs(-10)
# 10


