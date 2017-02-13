from __future__ import absolute_import
import numpy as np
import os
import sys
import _pickle as cPickle
from utils import maybe_download


def load_data():
    """Loads CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = maybe_download(dirname, origin, './dataset', untar=True)

    nb_train_samples = 50000

    x_train = np.zeros((nb_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((nb_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def load_batch(fpath):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.

    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d['labels']

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

