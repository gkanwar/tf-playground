import math
import numpy as np
import random
import tensorflow as tf
from tf_lib import *

prefix = '/data/d03/platypus/home/gurtej/interpolators/potts_2d/'
ensemble_path = 'b{:.2f}_h{:.2f}_n3_N50000_4_32.dat'

betas = [0.6, 0.63]
hs = [0.05, 0.10, 0.15]

N_ensemble = len(betas) * len(hs)

Ncfg_tot = 50000
Ncfg = 1000
Lx = 4
Lt = 32

all_data = tf.placeholder(tf.int32, (N_ensemble, Ncfg, Lx, Lt))
all_labels = []
real_data = []
for beta in betas:
    for h in hs:
        print((beta,h))
        # placeholder = tf.placeholder(np.int, (Ncfg, Lx, Lt))
        all_labels.append((beta, h))
        # all_data.append(placeholder)
        # hold truncated real data in memory
        fname = prefix + ensemble_path.format(beta, h)
        print('Loading {}'.format(fname))
        data = np.fromfile(fname, dtype=np.float64).astype(np.int)
        data = data.reshape(Ncfg_tot, Lx, Lt)[:Ncfg]
        real_data.append(data)
real_data = np.array(real_data)
all_labels = np.array(all_labels)
all_labels = tf.convert_to_tensor(all_labels)
dataset = tf.data.Dataset.from_tensor_slices((all_data, all_labels))
dataset = dataset.shuffle(buffer_size=N_ensemble).batch(N_ensemble)
it = dataset.make_initializable_iterator()
ensembles, labels = it.get_next()

op1 = tf.convert_to_tensor([1,0,0,0])
op2 = tf.convert_to_tensor([2,0,0,0])
# TODO: Placeholder for ops, and add assert?
# op1_momproj = []
# op2_momproj = []
# for dx in range(Lx):
#     op1_momproj.append(tf.roll(op1, shift=-dx, axis=0))
#     op2_momproj.append(tf.roll(op2, shift=-dx, axis=0))
# op1_momproj = tf.stack(*op1_momproj)
# op2_momproj = tf.stack(*op2_momproj)

op1_val = tf.cast(tf.mod(tf.reduce_sum(
    tf.multiply(ensembles, tf.reshape(op1, shape=[Lx,1])),
    axis=2 # x-axis
    ), 3), tf.float32)
op2_val = tf.cast(tf.mod(tf.reduce_sum(
    tf.multiply(ensembles, tf.reshape(op2, shape=[Lx,1])),
    axis=2 # x-axis,
    ), 3), tf.float32)

# point-like ops
op1_cos_val = tf.cos(tf.constant(2*math.pi/3)*op1_val)
op2_cos_val = tf.cos(tf.constant(2*math.pi/3)*op2_val)

# now make twopt
twopts = []
for dt in range(Lt):
    op1_cos_roll = tf.roll(op1_cos_val, shift=-dt, axis=2) # t-axis
    twopt = (
        tf.reduce_sum(op1_cos_roll * op2_cos_val, axis=(1,2))
        / tf.constant(Lt*Ncfg, dtype=tf.float32))
    twopts.append(twopt)

# for each dt and (beta,h) print evaluation of twopt
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(it.initializer, feed_dict={all_data: real_data})
    out = sess.run(twopts)
    for dt,twopt in enumerate(out):
        print("dt = {:d}".format(dt))
        print(twopt)
