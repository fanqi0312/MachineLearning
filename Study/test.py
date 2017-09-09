#hello word
import tensorflow as tf

hello = tf.constant("Hell!TensorFlow")
sess = tf.Session()
print(sess.run(hello))


#
import  numpy as np

#creat data
#0-10的随机数
x_data = np.random.rand(100).astype(np.float32)     # 5
#0.3-0.4的随机数
y_data = x_data*0.1 + 0.3       #0.8

### create tensorflow structure start ###
#初始值为-1~1的随机数
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))  # 3
#初始值为0
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases   #15


#优化器减少误差
loss = tf.reduce_mean(tf.square(y-y_data))  #14.2
#学习效率
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
### create tensorflow structure end ###

#激活
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))



#####################Session 会话控制###########################

import tensorflow as tf

matrixl = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

#矩阵乘法
product = tf.matmul(matrixl, matrix2)

#方法1 手动关闭
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

#方法2 循环后自动关闭session
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)


