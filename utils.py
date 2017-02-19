import os
import sys
import tarfile
import shutil
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def maybe_download(filename, origin, dst_dir, untar=False):
    """Download and extract the tarball from Alex's website."""

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if untar:
        untar_fpath = os.path.join(dst_dir, filename)
        filepath = untar_fpath + '.tar.gz'
    else:
        filepath = os.path.join(dst_dir, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(origin, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    if not os.path.exists(untar_fpath):
        print('Untaring file...')
        tfile = tarfile.open(filepath, 'r:gz')
        try:
            tfile.extractall(path=dst_dir)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(untar_fpath):
                if os.path.isfile(untar_fpath):
                    os.remove(untar_fpath)
                else:
                    shutil.rmtree(untar_fpath)
            raise
        tfile.close()

    return untar_fpath


def to_categorical(y, nb_classes=None):
    """Converts a class vector (integers) to binary class matrix (one-hot vectors).

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
        nb_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, train_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.

    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      train_placeholder: The training indicator placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder,
                                   train_placeholder, False)
        tp_ = sess.run(eval_correct, feed_dict=feed_dict)
        true_count += tp_

    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))
    return precision


def fill_feed_dict(data_set, images_pl, labels_pl, train_pl, train_val):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
      train_pl: The training indicator placeholder, from placeholder_inputs().

    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """

    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        train_pl: train_val,
    }
    return feed_dict
