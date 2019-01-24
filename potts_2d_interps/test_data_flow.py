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
    all_labels = [1,2,3,4]
    Ncfg_tot = 50
    Ncfg = 5
    Lx = 4
    Lt = 32
    ps = {}
    batch_size = 8
    datasets = []
    for label in all_labels:
        placeholder = tf.placeholder(tf.int32, (Ncfg_tot, Lx, Lt))
        ps[label] = placeholder
        dataset = tf.data.Dataset.from_tensor_slices(
            (np.full((Ncfg_tot,), label), placeholder))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=Ncfg_tot).batch(Ncfg)
        datasets.append(dataset)
    big_dataset = tf.data.experimental.sample_from_datasets(datasets)
    big_dataset = big_dataset.batch(batch_size)
    big_it = big_dataset.make_initializable_iterator()
    labels, ensembles = big_it.get_next()
    fake_ensembles = [
        np.random.randint(l*100, size=(Ncfg_tot, Lx, Lt)) for l in all_labels]
    feed_ensembles = {}
    for i,l in enumerate(all_labels):
        feed_ensembles[ps[l]] = fake_ensembles[i]
    with tf.Session() as sess:
        sess.run(big_it.initializer, feed_dict=feed_ensembles)
        for i in range(10):
            out_ls, out_ens = sess.run([labels, ensembles])
            print(out_ls[:,0])
            print(list(map(np.mean, out_ens)))
multi_ens_bootstrap()
