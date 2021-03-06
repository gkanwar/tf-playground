## Common functions within TF.

import tensorflow as tf

# Simple linear layer
def linear(inp, output_dim, scope, init_stddev=1.0):
    if init_stddev is not None:
        init = tf.random_normal_initializer(stddev=init_stddev)
    else:
        init = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        W = tf.get_variable(
            'W', [inp.get_shape()[1], output_dim],
            initializer=init)
        b = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(0.0))
        return tf.matmul(inp, W) + b

# Inverse sinh activation func
def isinh(z, scope):
    with tf.variable_scope(scope):
        return tf.log(z + tf.sqrt(z*z + 1))
def d_isinh(z):
    return tf.pow(tf.square(z*z + 1), -1)

# sinh activation func
def sinh(z, scope):
    with tf.variable_scope(scope):
        return (tf.exp(z) - tf.exp(-z))/2
def d_sinh(z):
    return (tf.exp(z) + tf.exp(-z))/2
