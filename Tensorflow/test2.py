import tensorflow as tf

#################张量
#tf.constant是一个计算，结果为一个张量，保存在a中

name = "1"

a = tf.constant([1.0, 2.0], name="a1")
# a = tf.constant([1, 2], name="a1") #如果是1,0，则会报类型错误
# a = tf.constant([1, 2], name="a1", dtype=tf.float32) #指定类型则不会报错
b = tf.constant([2.0, 3.0], name="b2")
result1 = tf.add(a, b, name="add3")
result2 = a + b

# 输出张量的结构，包含三个属性：名称:编号（从0开始）、维度（shape，长度为2）、类型（dtype）
print(result1)   #Tensor("add3:0", shape=(2,), dtype=float32)
print(result2)   #Tensor("add :0", shape=(2,), dtype=float32)




#################会话
"""
#1
#创建一个会话
sess = tf.Session()

sess.run(...)
#关闭会话
sess.close()

#2
#创建一个会话
with tf.Session() as sess:
    sess.run(...) #计算内容放在with中
#不需要关闭
"""




#当前默认的计算图。所以下面这个操作输出值为True
print(a.graph is tf.get_default_graph()) #True
#
# g1 = tf.Graph()
# with g1.as_default():
#     #
#     v = tf.get_variable(
#         "v",initializer=tf.zeros_initializer(shape=[1])
#     )
#
# g2 = tf.Graph()
# with g2.as_default():
#     #
#     v = tf.get_variable(
#         "v",initializer=tf.ones_initializer(shape=[1])
#     )
#
# with tf.Session(graph=g1) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable("v")))
#
# with tf.Session(graph=g2) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable("v")))


