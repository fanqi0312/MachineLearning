import tensorflow as tf
import numpy as np

# 展示方法
def tfprint(a):
    with tf.Session() as sess:
        result = sess.run(a)
        print(result)

    # sess = tf.Session()
    # result = sess.run(a)
    # print(result)
    # sess.close()


# 1.创建变量
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])

y = tf.matmul(w, x)


# 2.初始化
#variables have to be explicitly initialized before you can run Ops
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # 3.运行session
    sess.run(init_op)
    # 4.显示结果
    print(y) # Tensor("MatMul:0", shape=(1, 1), dtype=float32)



print("================================构造矩阵")
# 值为0的矩阵
a = tf.zeros([3, 4], tf.int32) # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
tfprint(a)


# 模仿矩阵维度，创建新矩阵
tensor = tf.Variable([[1, 2, 3], [4, 5, 6]])
a = tf.zeros_like(tensor) # [[0, 0, 0], [0, 0, 0]]
# tfprint(a)

# 值为1的矩阵
tf.ones([2, 3], tf.int32) # [[1, 1, 1], [1, 1, 1]]

# 'tensor' is [[1, 2, 3], [4, 5, 6]]
tf.ones_like(tensor) # [[1, 1, 1], [1, 1, 1]]

# 常量
# Constant 1-D Tensor populated with value list.
tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) # [1 2 3 4 5 6 7]
tfprint(tensor)

# Constant 2-D tensor populated with scalar value -1.
tensor = tf.constant(-1.0, shape=[2, 3]) # [[-1. -1. -1.],[-1. -1. -1.]]

#10-12，创建3个
tf.linspace(10.0, 12.0, 3, name="linspace") # [ 10.0  11.0  12.0]

#从3-18，每间隔3一个
# 'start' is 3
# 'limit' is 18
# 'delta' is 3
start = 3
limit = 18
delta = 3
a = tf.range(start, limit, delta) # [3, 6, 9, 12, 15]
tfprint(a)


norm = tf.random_normal([2, 3], mean=-1, stddev=4)
tfprint(norm)
# [[-1.86181378  1.07706356 -1.63136792]
#  [-5.19293594 -3.18992305 -0.6896944 ]]

# Shuffle the first dimension of a tensor
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)
tfprint(shuff)
# [[3 4]
#  [1 2]
#  [5 6]]


# 初始值为0，每次循环加1并打印
state = tf.Variable(0)
# 加法
new_value = tf.add(state, tf.constant(1))
# 赋值
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
# 0
# 1
# 2
# 3

print("================================saver.save   保存Session")
#tf.train.Saver
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
y = tf.matmul(w, x)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
# Do some work with the model.
# Save the variables to disk.
#     save_path = saver.save(sess, "C://tensorflow//model//test")
#     print("Model saved in file: ", save_path)


print("================================convert_to_tensor   Numpy 转化为 Tp（不建议）")
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
     print(sess.run(ta))


print("================================placeholder   预留空位置，再赋值")
# 只有数据格式，没值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    # 运行时赋值
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))


