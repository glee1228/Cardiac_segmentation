from keras import backend as K
import numpy as np
import tensorflow as tf
smooth = 1.

def average_dice_coef(y_true, y_pred):
    loss = 0
    label_length = y_pred.get_shape().as_list()[-1]
    print(label_length)
    for num_label in range(label_length):
        y_true_f = K.flatten(y_true[..., num_label])
        print(sess.run(y_true_f))
        y_pred_f = K.flatten(y_pred[..., num_label])
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss / label_length



y_true = tf.constant(1.,shape=[10,12])
y_true_static_shape = y_true.get_shape()
print(y_true_static_shape.as_list()[-1])

y_pred = tf.constant(1.,shape=[10,12])
y_pred_static_shape = y_pred.get_shape()
print(y_pred_static_shape.as_list()[-1])

with tf.Session() as sess:
    result=average_dice_coef(y_true,y_pred)
    print(sess.run(result))