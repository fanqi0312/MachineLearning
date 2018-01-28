"""
保存和加载训练好的模型

1. 创建目录（save目录）
2. 初始化Saver
3. 训练模型
4. 保存模型
5. 加载模型
6. 测试模型

"""
import tensorflow as tf

# 1. 创建目录（save目录）

# 开关：1训练保存，0测试加载
do_train = 0

v1 = tf.Variable(tf.random_normal([1,2]), name="v1")
v2 = tf.Variable(tf.random_normal([2,3]), name="v2")
init_op = tf.global_variables_initializer()
# 2. 初始化Saver
saver = tf.train.Saver()
with tf.Session() as sess:
    if do_train == 1:
        # 3. 训练模型
        sess.run(init_op)
        print ("V1:",sess.run(v1))
        print ("V2:",sess.run(v2))
        # 4. 保存模型
        saver_path = saver.save(sess, "model/modelSave.ckpt")
        print ("Model saved in file: ", saver_path)

    if do_train == 0:
        # 5. 加载模型
        saver.restore(sess, "model/modelSave.ckpt")

        # 6. 测试模型
        print ("V1:",sess.run(v1))
        print ("V2:",sess.run(v2))
        print ("Model restored")


print("================================ saver.save   保存Session")
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
