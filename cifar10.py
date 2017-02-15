# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading CIFAR-10 data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import numpy as np
import os
import sys
import _pickle as cPickle
from utils import maybe_download, to_categorical

NB_CLASSES = 10
# input image dimensions
IM_ROWS, IM_COLS = 32, 32
# The CIFAR10 images are RGB.
IM_CH = 3


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 dtype=dtypes.float32):
        """Construct a DataSet.
        `dtype` can be either `uint8` to leave the input as `[0, 255]`,
        or `float32` to rescale into `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        if one_hot:
            self._labels = to_categorical(self._labels, nb_classes=NB_CLASSES)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def load_data(dst_dir='./dataset'):
    """Loads CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = maybe_download(dirname, origin, dst_dir, untar=True)

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

    y_train = np.reshape(y_train, (len(y_train)))
    y_test = np.reshape(y_test, (len(y_test)))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def load_batch(fpath):
    """Internal utility for parsing CIFAR10 data.

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


def read_data_sets(dst_dir='./dataset', dtype=dtypes.float32, validation_size=5000):
    (train_images, train_labels), (test_images, test_labels) = load_data(dst_dir)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[-validation_size:]
    validation_labels = train_labels[-validation_size:]
    train_images = train_images[:-validation_size]
    train_labels = train_labels[:-validation_size]

    train = DataSet(train_images, train_labels, dtype=dtype, one_hot=False)
    validation = DataSet(validation_images,
                         validation_labels,
                         dtype=dtype, one_hot=False)
    test = DataSet(test_images, test_labels, dtype=dtype, one_hot=False)

    return base.Datasets(train=train, validation=validation, test=test)
