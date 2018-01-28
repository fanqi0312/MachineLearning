"""
Demo

"""
import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点. 0-10的随机数
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
# x_data = np.random.rand(100).astype(np.float32)

y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
# 初始值为-1~1的随机数
Weights = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
#
biases = tf.Variable(tf.zeros([1]))
y = tf.matmul(Weights, x_data) + biases


loss = tf.reduce_mean(tf.square(y - y_data))
# 优化器减少误差,学习效率0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 最小化方差
train = optimizer.minimize(loss)


# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]