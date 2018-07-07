#!/usr/bin/env python

## Toy problem: Approximately generate 1D Gaussian using GANs
## Source: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/

import numpy as np
import tensorflow as tf

# "Real" distribution to compare against
class DataDist(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N).astype(np.float32)
        samples.sort()
        return samples

# Random noise distribution for generator
class GeneratorDist(object):
    def __init__(self, width):
        self.width = width
        self.scale = 0.01

    def sample(self, N):
        return (
            np.linspace(-self.width, self.width, N)
            + np.random.random(N) * self.scale
        ).astype(np.float32)

# Simple linear layer
def linear(inp, output_dim, scope, init_stddev=1.0):
    with tf.variable_scope(scope):
        W = tf.get_variable(
            'W',
            [inp.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=init_stddev))
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0))
        return tf.matmul(inp, W) + b

# linear -> softplus (basically smooth ReLU) -> linear
def generator(inp, hidden_size):
    h0 = tf.nn.softplus(linear(inp, hidden_size, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1

# apparently better to have a deeper discriminator
# (linear -> tanh ->)^3 -> linear -> sigmoid (prob)
def discriminator(inp, hidden_size):
    h0 = tf.tanh(linear(inp, hidden_size*2, 'd0'))
    h1 = tf.tanh(linear(h0, hidden_size*2, 'd1'))
    h2 = tf.tanh(linear(h1, hidden_size*2, 'd2'))
    h3 = tf.sigmoid(linear(h2, 1, 'd3')) # need [0,1] prob
    return h3

## Network
batch_size = 8
hidden_size = 4
with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, shape=(batch_size,1))
    G = generator(z, hidden_size)

with tf.variable_scope('D') as scope:
    x = tf.placeholder(tf.float32, shape=(batch_size,1))
    # discriminator with the same variables hooked up to both
    # "real" and generated data
    D1 = discriminator(x, hidden_size)
    scope.reuse_variables()
    D2 = discriminator(G, hidden_size)

loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
loss_g = tf.reduce_mean(-tf.log(D2))

## Optimization (blatantly stealing the tuned params)
def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss, global_step=step, var_list=var_list)
    return optimizer

vs = tf.trainable_variables()
d_params = [v for v in vs if v.name.startswith('D/')]
g_params = [v for v in vs if v.name.startswith('G/')]

opt_d = optimizer(loss_d, d_params)
opt_g = optimizer(loss_g, g_params)

## Main
num_steps = 5000
data = DataDist()
gen = GeneratorDist(8.0)
with tf.Session() as session:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    for i in xrange(num_steps):
        # discriminator
        x_train = data.sample(batch_size)
        z_train = gen.sample(batch_size)
        ld, _ = session.run([loss_d, opt_d], {
            x: np.reshape(x_train, (batch_size, 1)),
            z: np.reshape(z_train, (batch_size, 1))
        })

        # generator
        z_train = gen.sample(batch_size)
        lg, _ = session.run([loss_g, opt_g], {
            z: np.reshape(z_train, (batch_size, 1))
        })

        if i % 10 == 0:
            print i, "\t", ld, "\t", lg
        
    print "Done!"
    n_test_batches = 100
    hreal,_ = np.histogram(data.sample(batch_size*n_test_batches), bins=50, range=(-10.0, 10.0))
    z_train = gen.sample(batch_size*n_test_batches)
    data = np.zeros((batch_size*n_test_batches, 1))
    for i in xrange(n_test_batches):
        z_batch = z_train[i*batch_size : (i+1)*batch_size]
        z_batch = np.reshape(z_batch, (batch_size, 1))
        data[i*batch_size : (i+1)*batch_size] = session.run(G, {z: z_batch})
    hfake,_ = np.histogram(data, bins=50, range=(-10.0, 10.0))
    print hreal
    print hfake
    print hreal - hfake
