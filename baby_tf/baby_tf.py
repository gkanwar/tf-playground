#!/usr/bin/env python

## Source: https://medium.com/all-of-us-are-belong-to-machines/the-gentlest-introduction-to-tensorflow-248dc871a224

import random
import numpy as np
import tensorflow as tf

## Network
x = tf.placeholder(tf.float32, [None, 1]) # input
W = tf.Variable(tf.zeros([1,1])) # transfer
b = tf.Variable(tf.zeros([1])) # bias
y = tf.matmul(x, W) + b # model output

## Model
yp = tf.placeholder(tf.float32, [None, 1]) # true output
cost = tf.reduce_sum(tf.pow((yp - y), 2)) # squared difference loss

## Training data (fake)
realW = 0.4
realb = 10.0
xs = []
ys = []
inds = range(100)
for i in range(100):
    xs.append([i])# + 0.2 * (random.random()-0.5)]])
    ys.append([realW * i + realb])# + 0.2 * (random.random()-0.5)]])
xs = np.array(xs)
ys = np.array(ys)

## Training model
learn_rate = 0.1
steps = 1000
batch_size = 10
# NOTE: It was key to use Adam optimizer with momentum to increase
# learn rate to a reasonable point.
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cost)

## Main
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(steps):
    batch_ind = i % (len(inds) // batch_size)
    # shuffle up data after each epoch
    if batch_ind == 0:
        random.shuffle(inds)
    # select a batch from the training set
    batch_xs = xs[inds[batch_ind : batch_ind + batch_size]]
    batch_ys = ys[inds[batch_ind : batch_ind + batch_size]]
    feed = { x: batch_xs, yp: batch_ys }
    sess.run(train_step, feed_dict=feed)

    print "Iter", i #, "x:", batch_xs, "y:", batch_ys
    print "W:", sess.run(W), "b:", sess.run(b)
    print "Cost:", sess.run(cost, feed_dict=feed)
