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

NDIM = 2

# "Real" 2D (!) distribution to compare against
class DataDist(object):
    def __init__(self):
        self.mean = np.array([4,2]).astype(np.float32)
        self.vs = np.array([0.5, 1.0]).astype(np.float32)
        theta = pi / 4
        rot = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
        rotT = np.transpose(rot)
        covDiag = np.array([[self.vs[0], 0.0], [0.0, self.vs[1]]])
        self.cov = np.dot(np.dot(rotT, covDiag), rot)
        self.norm = 1 / (2*pi*self.vs[0]*self.vs[1])

    def sample(self, N):
        samples = np.random.multivariate_normal(
            self.mean, self.cov, N).astype(np.float32)
        return samples

    def make_p(self, tf_out):
        return self.norm * tf.exp(-tf.tensordot(
            (tf_out - self.mean)**2, (2*self.vs**2)**(-1),
            axes = [[-1], [0]]))

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
        self.mean = (0,0)
        self.vs = np.array([sigma, sigma])
        self.cov = [[self.vs[0], 0.0], [0.0, self.vs[1]]]
        self.norm = 1 / (2*pi*self.vs[0]*self.vs[1])

    def sample(self, N):
        samples = np.random.multivariate_normal(self.mean, self.cov, N).astype(np.float32)
        pdens = self.norm * np.exp(-np.dot(np.square(samples), (2*self.vs**2)**(-1))).astype(np.float32)
        return (samples, pdens)

class UniformNoiseDist(NoiseDist):
    def __init__(self, sigma):
        self.min = -sigma
        self.max = sigma
        self.norm1d = 1 / (2*sigma)

    def sample(self, N):
        samples = np.random.uniform(self.min, self.max, (N,NDIM)).astype(np.float32)
        pdens = np.full(N, self.norm1d**2).astype(np.float32)
        return (samples, pdens)

# Generator is a simple
# [ linear -> sinh -> linear]
# network. Note that the actual function should just be linear in the input
# distribution, so the network should learn to push the input through the linear
# regime of the sinh.
class Generator(object):
    def __init__(self, inp):
        self.inp = inp
        # Make forward pass
        with tf.variable_scope('forward'):
            self.g0 = linear(inp, inp.get_shape()[1], 'g0')
            self.g1 = sinh(self.g0, 'g1')
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
            self.d_g1 = d_sinh(self.g0)
            self.d_g2 = self.g2_W
            # TODO: Unclear if determinint does the right thing when batching!!
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
        self.eta = tf.placeholder(tf.float32, shape=(batch_size, NDIM))
        self.p_eta = tf.placeholder(tf.float32, shape=(batch_size, 1))
        with tf.variable_scope('G'):
            self.G = Generator(self.eta)
        with tf.variable_scope('L'):
            self.p_out = data_dist.make_p(self.G.out)
            # Loss fn is squared diff of logs
            self.loss_g = tf.reduce_mean(
                (tf.log(self.p_out) + tf.log(self.G.detJ) - tf.log(self.p_eta))**2)
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
            loss, detJ, p_out, g2_b, _, = session.run([net.loss_g, net.G.detJ, net.p_out, net.G.g2_b, net.opt_g], {
                net.eta: np.reshape(eta_train, (batch_size, NDIM)),
                net.p_eta: np.reshape(p_eta_train, (batch_size, 1))
            })

            if i % args.print_freq == 0:
                print i, "\t", loss, detJ, p_out, p_eta_train, g2_b
        print "Done!"
        # Evaluate result
        n_test_batches = 1000
        real_data = data.sample(batch_size*n_test_batches)
        hreal_x,bins_real_x = np.histogram(
            real_data[:,0], bins=50, range=(-10.0, 10.0))
        hreal_y,bins_real_y = np.histogram(
            real_data[:,1], bins=50, range=(-10.0, 10.0))
        eta_train, p_eta_train = noise.sample(batch_size*n_test_batches)
        data = np.zeros((batch_size*n_test_batches, NDIM))
        for i in xrange(n_test_batches):
            eta_batch = eta_train[i*batch_size : (i+1)*batch_size]
            eta_batch = np.reshape(eta_batch, (batch_size, NDIM))
            p_eta_batch = p_eta_train[i*batch_size : (i+1)*batch_size]
            p_eta_batch = np.reshape(p_eta_batch, (batch_size, 1))
            data[i*batch_size : (i+1)*batch_size] = session.run(
                net.G.out, {
                    net.eta: eta_batch,
                    net.p_eta: p_eta_batch
                })
        hfake_x,bins_fake_x = np.histogram(
            data[:,0], bins=50, range=(-10.0, 10.0))
        hfake_y,bins_fake_y = np.histogram(
            data[:,1], bins=50, range=(-10.0, 10.0))
        print hreal_x
        print hfake_x
        print hreal_x-hfake_x
        # Also plot?
        if not _headless:
            fig, axes = plt.subplots(2)
            axes[0].hist2d(real_data[:,0], real_data[:,1], bins=20, range=[[1.0, 7.0], [-1.0, 5.0]])
            axes[1].hist2d(data[:,0], data[:,1], bins=20, range=[[1.0, 7.0], [-1.0, 5.0]])
            # fig, axes = plt.subplots(2, 2)
            # width = np.diff(bins_real_x)
            # center = (bins_real_x[:-1] + bins_real_x[1:]) / 2.0
            # axes[0,0].bar(center, hreal_x, align='center', width=width)
            # width = np.diff(bins_fake_x)
            # center = (bins_fake_x[:-1] + bins_fake_x[1:]) / 2.0
            # axes[1,0].bar(center, hfake_x, align='center', width=width)
            # width = np.diff(bins_real_y)
            # center = (bins_real_y[:-1] + bins_real_y[1:]) / 2.0
            # axes[0,1].bar(center, hreal_y, align='center', width=width)
            # width = np.diff(bins_fake_y)
            # center = (bins_fake_y[:-1] + bins_fake_y[1:]) / 2.0
            # axes[1,1].bar(center, hfake_y, align='center', width=width)
            fig.savefig('./toddler_dist.png')
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Learn a distribution using Jacobian method.')
    # Core args
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--learn_rate', type=float, default=0.01)
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('--input_noise_type', type=str,
                        default=NoiseDist.types[0], choices=NoiseDist.types)
    parser.add_argument('--input_noise_width', type=float, default=1.0)
    # Candy
    parser.add_argument('--print_freq', type=int, default=10)
    args = parser.parse_args()
    main(args)
