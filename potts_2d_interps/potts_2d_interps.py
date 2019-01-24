import math
import numpy as np
import random
import tensorflow as tf
import time

from tf_lib import *

prefix = '/data/d03/platypus/home/gurtej/interpolators/potts_2d/'
tf_prefix = prefix + 'tf_model/'
tf_old_prefix = prefix + 'tf_model_old/'
ensemble_path = 'b{:.2f}_h{:.2f}_n3_N50000_4_32.dat'

all_betas = [0.6, 0.63]
all_hs = [0.05, 0.10, 0.15]
all_params = []
for beta in all_betas:
    for h in all_hs:
        all_params.append((beta,h))

Ncfg_tot = 50000
Ncfg = 100
Lx = 4
Lt = 32
batch_size = 64
hidden_size = 16
embed_size = 16
learn_rate = 1e-1
learn_decay_rate = 0.9
learn_decay_steps = 1000
num_iters = 100000
adam_eps = 1e-6

def bootstrap_dataset(t, boot_size, labels):
    Ncfg = t.shape[0]
    labels = [tf.constant(np.full((Ncfg,), label), dtype=tf.float32)
              for label in labels]
    dataset = tf.data.Dataset.from_tensor_slices(tuple(labels + [t]))
    dataset = dataset.repeat().shuffle(buffer_size=Ncfg).batch(boot_size)
    return dataset

def bootstrap_multi_ensemble(all_params, shape, boot_size, batch_size):
    inputs = {}
    datasets = []
    for param in all_params:
        print(param)
        # build bootstrapped pipeline
        placeholder = tf.placeholder(tf.int32, shape)
        inputs[tuple(param)] = placeholder
        dataset = bootstrap_dataset(placeholder, boot_size, list(param))
        datasets.append(dataset)
    big_dataset = tf.data.experimental.sample_from_datasets(datasets)
    big_dataset = big_dataset.batch(batch_size)
    big_it = big_dataset.make_initializable_iterator()
    return big_it, inputs

## Establish bootstrap and data feed
big_it, inputs = bootstrap_multi_ensemble(
    all_params, (Ncfg_tot, Lx, Lt), Ncfg, batch_size)
betas, hs, ensembles = big_it.get_next()
real_data = {}
feed_ensembles = {}
for beta,h in all_params:
    # load data into mem
    print((beta,h))
    fname = prefix + ensemble_path.format(beta, h)
    print('Loading {}'.format(fname))
    data = np.fromfile(fname, dtype=np.float64).astype(np.int)
    data = data.reshape(Ncfg_tot, Lx, Lt).astype(np.float32)
    real_data[(beta,h)] = data
    feed_ensembles[inputs[(beta,h)]] = real_data[(beta,h)]

## Build evaluation network as noisy ground truth
op1 = tf.convert_to_tensor([1,0,0,0])
op2 = tf.convert_to_tensor([2,0,0,0])
# TODO: Placeholder for ops, and add assert?
op1_momproj = []
op2_momproj = []
for dx in range(Lx):
    op1_momproj.append(tf.roll(op1, shift=-dx, axis=0))
    op2_momproj.append(tf.roll(op2, shift=-dx, axis=0))
op1_momproj = tf.stack(op1_momproj)
op2_momproj = tf.stack(op2_momproj)
print('op1_momproj.shape = {}'.format(op1_momproj.shape))
print('ensembles.shape = {}'.format(ensembles.shape))
print('betas.shape = {}'.format(betas.shape))

op1_val = tf.cast(tf.mod(
    tf.tensordot(ensembles, op1_momproj, axes=[[2], [1]])
    , 3), tf.float32)
op2_val = tf.cast(tf.mod(
    tf.tensordot(ensembles, op2_momproj, axes=[[2], [1]])
    , 3), tf.float32)

# point-like ops
op1_cos_val = tf.cos(tf.constant(2*math.pi/3)*op1_val)
op2_cos_val = tf.cos(tf.constant(2*math.pi/3)*op2_val)
op1_cos_val = tf.reduce_sum(op1_cos_val, axis=3)
op2_cos_val = tf.reduce_sum(op2_cos_val, axis=3)

# now make twopt
twopts = []
for dt in range(Lt):
    op1_cos_roll = tf.roll(op1_cos_val, shift=-dt, axis=2) # t-axis
    twopt = (
        tf.reduce_sum(op1_cos_roll * op2_cos_val, axis=(1,2))
        / tf.constant(Lt*Ncfg, dtype=tf.float32))
    twopts.append(twopt)

# build embedding layers
op1_embed = tf.tile(op1, [3])[Lx-1:2*Lx+1]
op1_embed = tf.tile(tf.reshape(op1_embed, [1,Lx+2,1]), [batch_size, 1, 1])
op1_embed = tf.cast(op1_embed, tf.float32)
op2_embed = tf.tile(op2, [3])[Lx-1:2*Lx+1]
op2_embed = tf.tile(tf.reshape(op2_embed, [1,Lx+2,1]), [batch_size, 1, 1])
op2_embed = tf.cast(op2_embed, tf.float32)
with tf.variable_scope('embed'):
    kernel1 = tf.get_variable(
        'kernel1', shape=[3, 1, hidden_size // 4], dtype=np.float32,
        initializer=tf.random_normal_initializer())
    kernel2 = tf.get_variable(
        'kernel2', shape=[3, hidden_size // 4, hidden_size], dtype=np.float32,
        initializer=tf.random_normal_initializer())
op1_conv = tf.nn.conv1d(op1_embed, kernel1, 1, 'VALID')
op1_conv = tf.nn.conv1d(op1_conv, kernel2, 2, 'VALID')
op2_conv = tf.nn.conv1d(op2_embed, kernel1, 1, 'VALID')
op2_conv = tf.nn.conv1d(op2_conv, kernel2, 2, 'VALID')
betas_embed = tf.slice(betas, [0,0], [batch_size, 1])
hs_embed = tf.slice(hs, [0,0], [batch_size, 1])
inp1 = tf.concat([tf.squeeze(op1_conv), betas_embed, hs_embed], 1)
inp2 = tf.concat([tf.squeeze(op2_conv), betas_embed, hs_embed], 1)
out1 = linear(inp1, embed_size, 'embed_g0')
out2 = linear(inp2, embed_size, 'embed_g0') # reuse
out = tf.reduce_sum(tf.multiply(out1, out2), axis=1)

loss = tf.reduce_mean(tf.squared_difference(twopts[1], out))

global_step = tf.Variable(0, trainable=False)
learn_rate_node = tf.train.exponential_decay(
    learn_rate, global_step, learn_decay_steps, learn_decay_rate, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate=learn_rate_node, epsilon=adam_eps)
train_op = opt.minimize(loss, global_step=global_step)

saver = tf.train.Saver()

## Do the thing!
start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, tf_old_prefix + 'tf_model.final')
    sess.run(big_it.initializer, feed_dict=feed_ensembles)
    loss_history = []
    for i in range(num_iters):
        _,l,twopt1,fake_twopt1 = sess.run([train_op, loss, twopts[1], out])
        loss_history.append(l)
        if i % 100 == 0:
            print('Iter {} ({:.1f}s): loss = {}'.format(i, time.time()-start, l))
            print(twopt1)
            print(fake_twopt1)
        if i % 1000 == 0: # save the model
            fname = saver.save(sess, tf_prefix + 'tf_model.iter_{:d}'.format(i))
            print('Saved model in {}'.format(fname))
        # print(big_out[Lt][:,0]) # betas
        # print(big_out[Lt+1][:,0]) # hs
        # print(big_out[Lt+2]) # inner prod outputs
        # for dt,twopt in enumerate(big_out[:Lt]):
        #     print("dt = {:d}".format(dt))
        #     print(twopt)
    fname = saver.save(sess, tf_prefix + 'tf_model.final')
    print('Saved final model in {}'.format(fname))
    print('Training COMPLETED in {:.1f}s'.format(time.time() - start))
    fname = tf_prefix + 'tf_loss.dat'
    np.array(loss_history).tofile(fname)
    print('Wrote loss history to {}'.format(fname))
