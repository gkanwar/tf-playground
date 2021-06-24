#!/usr/bin/env python

## Toy problem: Approximately generate 1D Gaussian using Jacobian method.
## Train using real distribution samples for faster convergence?

import argparse
from math import *
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
        self.norm = 1 / (sqrt(2*pi)*self.sigma)

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N).astype(np.float32)
        samples.sort()
        return samples

    def make_p(self, tf_out):
        return self.norm * tf.exp(-(tf_out - self.mu)**2 / (2*self.sigma**2))

# Random gaussian noise distribution for generator.
class NoiseDist:
    types = ('gaussian', 'uniform')
    @staticmethod
    def get_cls(type):
        if type == 'gaussian':
            return GaussNoiseDist
        elif type == 'uniform':
            return UniformNoiseDist

class GaussNoiseDist(NoiseDist):
    def __init__(self, sigma):
        self.mu = 0.0
        self.sigma = sigma
        self.norm = 1 / (sqrt(2*pi)*sigma)

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N).astype(np.float32)
        pdens = self.norm * np.exp(-np.square(samples) / (2*self.sigma**2)).astype(np.float32)
        return (samples, pdens)

class UniformNoiseDist(NoiseDist):
    def __init__(self, sigma):
        self.min = -sigma
        self.max = sigma
        self.norm = 1 / (2*sigma)

    def sample(self, N):
        samples = np.random.uniform(self.min, self.max, N).astype(np.float32)
        pdens = np.full(N, self.norm).astype(np.float32)
        return (samples, pdens)

# Generator is a simple
# [ linear -> isinh -> linear ]
# network. Note that the actual function should just be linear in the input
# distribution, so the network should learn to push the input through the linear
# regime of the isinh.
class Generator(object):
    def __init__(self, inp):
        self.inp = inp
        # Make forward pass
        with tf.variable_scope('forward'):
            self.g0 = linear(inp, inp.get_shape()[1], 'g0')
            self.g1 = isinh(self.g0, 'g1')
            # with tf.variable_scope('h0', reuse=True):
            #     self.h0_b = tf.get_variable('b')
            self.g2 = linear(self.g1, self.g1.get_shape()[1], 'g2')
            self.out = self.g2
            with tf.variable_scope('g0', reuse=True):
                self.g0_W = tf.get_variable('W')
                self.g0_b = tf.get_variable('b')
            with tf.variable_scope('g2', reuse=True):
                self.g2_W = tf.get_variable('W')
                self.g2_b = tf.get_variable('b')
        # Make detJ pass
        with tf.variable_scope('detJ'):
            self.d_g0 = self.g0_W
            self.d_g1 = d_isinh(self.g0)
            self.d_g2 = self.g2_W
            self.detJ = tf.abs(tf.reduce_prod(self.d_g1) *
                               tf.matrix_determinant(self.d_g0) *
                               tf.matrix_determinant(self.d_g2))

# Adam optimizer
def make_optimizer(loss, var_list, learn_rate):
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(
        loss, global_step=step, var_list=var_list)
    return optimizer

class Network(object):
    def __init__(self, data_dist, batch_size, learn_rate):
        self.eta = tf.placeholder(tf.float32, shape=(batch_size, 1))
        self.p_eta = tf.placeholder(tf.float32, shape=(batch_size, 1))
        with tf.variable_scope('G'):
            self.G = Generator(self.eta)
        with tf.variable_scope('L'):
            self.p_out = data_dist.make_p(self.G.out)
            self.loss_det = self.p_out * self.G.detJ / self.p_eta
            self.loss_ent = tf.log(self.p_out / self.p_eta)
            # # Loss fn is squared Kullback-Leibler divergence est
            # self.loss_g = tf.reduce_mean(tf.pow(
            #     self.loss_det * self.loss_ent, 2))
            # Loss fn is squared det matching criterion
            self.loss_g = tf.reduce_mean(tf.pow(
                tf.log(self.loss_det), 2))
        vs = tf.trainable_variables()
        self.opt_g = make_optimizer(self.loss_g, vs, learn_rate)

def main(args):
    batch_size = args.batch_size
    data = DataDist()
    net = Network(data, batch_size, args.learn_rate)
    
    noise = NoiseDist.get_cls(args.input_noise_type)(args.input_noise_width)
    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        for i in xrange(args.num_steps):
            eta_train, p_eta_train = noise.sample(batch_size)
            loss, _ = session.run([net.loss_g, net.opt_g], {
                net.eta: np.reshape(eta_train, (batch_size, 1)),
                net.p_eta: np.reshape(p_eta_train, (batch_size, 1))
            })

            if i % args.print_freq == 0:
                print i, "\t", loss, session.run(net.G.g0_b), session.run(net.G.g2_b)
        print "Done!"
        # Evaluate result
        n_test_batches = 100
        hreal,bins_real = np.histogram(
            data.sample(batch_size*n_test_batches), bins=50, range=(-10.0, 10.0))
        eta_train, p_eta_train = noise.sample(batch_size*n_test_batches)
        data = np.zeros((batch_size*n_test_batches, 1))
        for i in xrange(n_test_batches):
            eta_batch = eta_train[i*batch_size : (i+1)*batch_size]
            eta_batch = np.reshape(eta_batch, (batch_size, 1))
            p_eta_batch = p_eta_train[i*batch_size : (i+1)*batch_size]
            p_eta_batch = np.reshape(p_eta_batch, (batch_size, 1))
            data[i*batch_size : (i+1)*batch_size] = session.run(
                net.G.out, {
                    net.eta: eta_batch,
                    net.p_eta: p_eta_batch
                })
        hfake,bins_fake = np.histogram(
            data, bins=50, range=(-10.0, 10.0))
        print hreal
        print hfake
        print hreal-hfake
        # Also plot?
        if not _headless:
            fig, axes = plt.subplots(2)
            width = np.diff(bins_real)
            center = (bins_real[:-1] + bins_real[1:]) / 2.0
            axes[0].bar(center, hreal, align='center', width=width)
            width = np.diff(bins_fake)
            center = (bins_fake[:-1] + bins_fake[1:]) / 2.0
            axes[1].bar(center, hfake, align='center', width=width)
            fig.savefig('./baby_dist.png')
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Learn a distribution using Jacobian method.')
    # Core args
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--learn_rate', type=float, default=0.01)
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('--input_noise_type', type=str,
                        default=NoiseDist.types[0], choices=NoiseDist.types)
    parser.add_argument('--input_noise_width', type=float, default=1.0)
    # Candy
    parser.add_argument('--print_freq', type=int, default=10)
    args = parser.parse_args()
    main(args)
