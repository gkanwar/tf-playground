import argparse
import math
import numpy as np
import random
import tensorflow as tf
import time

from tf_lib import *

prefix = '/data/d03/platypus/home/gurtej/interpolators/potts_2d/'
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
def build_embed(op, betas, hs, hidden_size):
    batch_size = op.shape[0]
    Lx = op.shape[1]
    op_embed = tf.tile(op, [1,3])[:,Lx-1:2*Lx+1]
    assert(op_embed.shape == (batch_size, Lx+2))
    op_embed = tf.reshape(op_embed, [batch_size,Lx+2,1])
    op_embed = tf.cast(op_embed, tf.float32)
    with tf.variable_scope('embed', reuse=tf.AUTO_REUSE):
        kernel1 = tf.get_variable(
            'kernel1', shape=[3, 1, hidden_size // 4], dtype=np.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        kernel2 = tf.get_variable(
            'kernel2', shape=[3, hidden_size // 4, hidden_size], dtype=np.float32,
            initializer=tf.contrib.layers.xavier_initializer())
    op_conv = tf.nn.conv1d(op_embed, kernel1, 1, 'VALID')
    op_conv = tf.nn.relu(op_conv)
    op_conv = tf.nn.conv1d(op_conv, kernel2, 2, 'VALID')
    op_conv = tf.nn.relu(op_conv)
    inp = tf.concat([tf.squeeze(op_conv), betas, hs], 1)
    out = linear(inp, embed_size, 'embed_g0')
    return out

betas_embed = tf.slice(betas, [0,0], [batch_size, 1])
hs_embed = tf.slice(hs, [0,0], [batch_size, 1])
# TODO: actually sample a stream of ops
op1_batch = tf.tile(tf.reshape(op1, [1,Lx]), [batch_size, 1])
op2_batch = tf.tile(tf.reshape(op2, [1,Lx]), [batch_size, 1])
out1 = build_embed(op1_batch, betas_embed, hs_embed, hidden_size)
out2 = build_embed(op2_batch, betas_embed, hs_embed, hidden_size)
out = tf.reduce_sum(tf.multiply(out1, out2), axis=1)

# overall loss and training
loss = tf.reduce_mean(tf.squared_difference(twopts[1], out))
global_step = tf.Variable(0, trainable=False)
learn_rate_node = tf.train.exponential_decay(
    learn_rate, global_step, learn_decay_steps, learn_decay_rate, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate=learn_rate_node, epsilon=adam_eps)
train_op = opt.minimize(loss, global_step=global_step)

# writing to file
saver = tf.train.Saver()



## Do the thing!
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=True)
    parser.add_argument('--model_subpath', type=str, required=True)
    args = parser.parse_args()
    print('Running with args = {}'.format(args))
    tf_prefix = prefix + args.model_subpath
    try: os.mkdir(tf_prefix)
    except: pass
    if args.action == 'train':
        start = time.time()
        print('Training model for {:d} iters.'.format(num_iters))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
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
    elif args.action == 'evaluate':
        start = time.time()
        print('Evaluating model on all params:')
        print(all_params)
        batch_size = len(all_params)
        op1_batch = tf.tile(tf.reshape(op1, [1,Lx]), [batch_size, 1])
        op2_batch = tf.tile(tf.reshape(op2, [1,Lx]), [batch_size, 1])
        eval_betas = []
        eval_hs = []
        for beta,h in all_params:
            eval_betas.append(beta)
            eval_hs.append(h)
        betas_embed = tf.constant(eval_betas, dtype=tf.float32)
        betas_embed = tf.reshape(betas_embed, [batch_size, 1])
        hs_embed = tf.constant(eval_hs, dtype=tf.float32)
        hs_embed = tf.reshape(hs_embed, [batch_size, 1])
        out1 = build_embed(op1_batch, betas_embed, hs_embed, hidden_size)
        out2 = build_embed(op2_batch, betas_embed, hs_embed, hidden_size)
        out = tf.reduce_sum(tf.multiply(out1, out2), axis=1)
        with tf.Session() as sess:
            saver.restore(sess, tf_prefix + 'tf_model.final')
            fake_twopt1 = sess.run([out])[0]
            print('Generated twopt {}'.format(fake_twopt1))
    else: assert(False)
