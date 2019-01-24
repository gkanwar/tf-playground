import math
import numpy as np
import random
import tensorflow as tf
from tf_lib import *

prefix = '/data/d03/platypus/home/gurtej/interpolators/potts_2d/'
ensemble_path = 'b{:.2f}_h{:.2f}_n3_N50000_4_32.dat'

all_betas = [0.6, 0.63]
all_hs = [0.05, 0.10, 0.15]

N_ensemble = len(all_betas) * len(all_hs)

Ncfg_tot = 50000
Ncfg = 1000
Lx = 4
Lt = 32
batch_size = 16

def bootstrap_dataset(t, boot_size, labels):
    Ncfg = t.shape[0]
    labels = [np.full((Ncfg,), label) for label in labels]
    dataset = tf.data.Dataset.from_tensor_slices(tuple(labels + [t]))
    dataset = dataset.repeat().shuffle(buffer_size=Ncfg).batch(boot_size)
    return dataset

inputs = {}
real_data = {}
datasets = []
for beta in all_betas:
    for h in all_hs:
        print((beta,h))
        # build bootstrapped pipeline
        placeholder = tf.placeholder(tf.int32, (Ncfg_tot, Lx, Lt))
        inputs[(beta,h)] = placeholder
        dataset = bootstrap_dataset(placeholder, Ncfg, [beta,h])
        datasets.append(dataset)
        # load data into mem
        fname = prefix + ensemble_path.format(beta, h)
        print('Loading {}'.format(fname))
        data = np.fromfile(fname, dtype=np.float64).astype(np.int)
        data = data.reshape(Ncfg_tot, Lx, Lt)
        real_data[(beta,h)] = data
big_dataset = tf.data.experimental.sample_from_datasets(datasets)
big_dataset = big_dataset.batch(batch_size)
big_it = big_dataset.make_initializable_iterator()
betas, hs, ensembles = big_it.get_next()

feed_ensembles = {}
for beta in all_betas:
    for h in all_hs:
        feed_ensembles[inputs[(beta,h)]] = real_data[(beta,h)]
        
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
    sess.run(big_it.initializer, feed_dict=feed_ensembles)
    out = sess.run(twopts + [betas, hs])
    print(out[-2][:,0]) # betas
    print(out[-1][:,0]) # hs
    for dt,twopt in enumerate(out[:Lt]):
        print("dt = {:d}".format(dt))
        print(twopt)
