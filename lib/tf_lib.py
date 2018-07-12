## Common functions within TF.

import tensorflow as tf

# Simple linear layer
def linear(inp, output_dim, scope, init_stddev=1.0):
    with tf.variable_scope(scope):
        W = tf.get_variable(
            'W', [inp.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=init_stddev))
        b = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(0.0))
        return tf.matmul(inp, W) + b

# Inverse sinh activation func
def isinh(z, scope):
    with tf.variable_scope(scope):
        off = tf.get_variable(
            'b', [z.get_shape()[1]],
            initializer=tf.constant_initializer(0.0))
        return tf.log(z + tf.sqrt(z*z + 1)) + off
def d_isinh(z):
    return tf.pow(tf.square(z*z + 1), -1)
