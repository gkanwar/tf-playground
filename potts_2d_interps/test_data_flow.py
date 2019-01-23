import numpy as np
import tensorflow as tf

## Single bootstrap example
def single_bootstrap():
    source = np.array([1,2,3,4])
    placeholder = tf.placeholder(tf.int32, (4,))
    dataset = tf.data.Dataset.from_tensor_slices(placeholder)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=4).batch(4)
    it = dataset.make_initializable_iterator()
    data = it.get_next()

    with tf.Session() as sess:
        sess.run(it.initializer, feed_dict={placeholder: source})
        for i in range(10):
            print(sess.run(data))

## Multi-ensemble bootstrap example
def multi_ens_bootstrap():
    labels = [1,2,3,4]
    Ncfg_tot = 50
    Ncfg = 1
    Lx = 4
    Lt = 32
    boots = []
    its = []
    big_dataset = None
    ps = {}
    for label in labels:
        placeholder = tf.placeholder(tf.int32, (Ncfg_tot, Lx, Lt))
        ps[label] = placeholder
        dataset = tf.data.Dataset.from_tensor_slices(placeholder)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=Ncfg_tot).batch(Ncfg)
        it = dataset.make_initializable_iterator()
        its.append(it)
        boot = it.get_next()
        boots.append((label, boot))
        if big_dataset is None:
            big_dataset = tf.data.Dataset.from_tensors((label, boot))
        else:
            big_dataset = big_dataset.concatenate(
                tf.data.Dataset.from_tensors((label, boot)))
    big_dataset = big_dataset.repeat()
    big_dataset = big_dataset.shuffle(buffer_size=len(labels)).batch(8)
    big_it = big_dataset.make_initializable_iterator()
    label_and_ens = big_it.get_next()
    fake_ensembles = [
        np.full((Ncfg_tot, Lx, Lt), l) for l in labels]
    feed_ensembles = {}
    for i,l in enumerate(labels):
        feed_ensembles[ps[l]] = fake_ensembles[i]
    with tf.Session() as sess:
        sess.run(list(map(lambda x: x.initializer, its)),
                 feed_dict = feed_ensembles)
        sess.run(big_it.initializer)
        for i in range(10):
            out = sess.run(label_and_ens)
            print(out[0])
