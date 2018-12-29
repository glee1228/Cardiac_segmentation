indices = [1, 2, 1, 3, 1, 1]
depth = 3
indices2 = [[[1, 2, 3],[4, 5, 6],[7,8,9]],[[1,0,0],[2,0,0],[3,0,0]]]
import tensorflow as tf
import numpy as np
import os




data_dir='/home/pirl/Downloads/cardiac/data/CHD_dia'
train_data_dir = os.path.join(data_dir,'image')

print(train_data_dir)

















# with tf.Session() as sess:
#     v = tf.Variable(indices2)
#     init = tf.initialize_all_variables()
#     #print(sess.run(tf.one_hot(indices, depth)))
#     #print(sess.run(tf.shape(tf.squeeze(indices))))
#     sess.run(init)
#     print(sess.run(v))
#     print(sess.run(tf.transpose(v, perm=[0, 2, 1])))
#     #print(sess.run(tf.transpose(indices2,[2,3])))
#     sess.close()