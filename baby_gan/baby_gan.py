#!/usr/bin/env python

## Toy problem: Approximately generate 1D Gaussian using GANs
## Source: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/

import argparse
import numpy as np
import tensorflow as tf
import os
import matplotlib
_headless = os.getenv('DISPLAY', '') == ''
if _headless: matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../lib/')
from tf_lib import *

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

## Optimization (blatantly stealing the tuned params)
def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss, global_step=step, var_list=var_list)
    return optimizer

## Network
class Network(object):
    def __init__(self, batch_size, hidden_size):
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(batch_size,1))
            self.G = generator(self.z, hidden_size)

        with tf.variable_scope('D') as scope:
            self.x = tf.placeholder(tf.float32, shape=(batch_size,1))
            # discriminator with the same variables hooked up to both
            # "real" and generated data
            self.D1 = discriminator(self.x, hidden_size)
            scope.reuse_variables()
            self.D2 = discriminator(self.G, hidden_size)

        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        vs = tf.trainable_variables()
        d_params = [v for v in vs if v.name.startswith('D/')]
        g_params = [v for v in vs if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, d_params)
        self.opt_g = optimizer(self.loss_g, g_params)

## Main
def main(args):
    batch_size = args.batch_size
    net = Network(batch_size, args.hidden_size)
    data = DataDist()
    gen = GeneratorDist(args.input_noise_range)
    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        for i in xrange(args.num_steps):
            # discriminator
            x_train = data.sample(batch_size)
            z_train = gen.sample(batch_size)
            ld, _ = session.run([net.loss_d, net.opt_d], {
                net.x: np.reshape(x_train, (batch_size, 1)),
                net.z: np.reshape(z_train, (batch_size, 1))
            })

            # generator
            z_train = gen.sample(batch_size)
            lg, _ = session.run([net.loss_g, net.opt_g], {
                net.z: np.reshape(z_train, (batch_size, 1))
            })

            if i % args.print_freq == 0:
                print i, "\t", ld, "\t", lg

        print "Done!"
        n_test_batches = 100
        hreal,bins_real = np.histogram(
            data.sample(batch_size*n_test_batches), bins=50, range=(-10.0, 10.0))
        z_train = gen.sample(batch_size*n_test_batches)
        data = np.zeros((batch_size*n_test_batches, 1))
        for i in xrange(n_test_batches):
            z_batch = z_train[i*batch_size : (i+1)*batch_size]
            z_batch = np.reshape(z_batch, (batch_size, 1))
            data[i*batch_size : (i+1)*batch_size] = session.run(net.G, {net.z: z_batch})
        hfake,bins_fake = np.histogram(
            data, bins=50, range=(-10.0, 10.0))
        print hreal
        print hfake
        print hreal - hfake
        # Also plot?
        if not _headless:
            fig, axes = plt.subplots(2)
            width = np.diff(bins_real)
            center = (bins_real[:-1] + bins_real[1:]) / 2.0
            axes[0].bar(center, hreal, align='center', width=width)
            width = np.diff(bins_fake)
            center = (bins_fake[:-1] + bins_fake[1:]) / 2.0
            axes[1].bar(center, hfake, align='center', width=width)
            fig.savefig('./baby_gan.png')
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baby GAN test')
    # Core args
    parser.add_argument('--num_steps', type=int, default=5000)
    parser.add_argument('--input_noise_range', type=float, default=8.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('-hs', '--hidden_size', type=int, default=4)
    # Candy
    parser.add_argument('--print_freq', type=int, default=10)
    args = parser.parse_args()
    main(args)
