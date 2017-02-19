#
# BINARY-CONNECT
# ==============
# The code was built on top of MNIST tutorials in Tensorflow GitHub repository:
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist
#
# It implements the paper BinaryConnect: Training Deep Neural Networks with binary weights during propagations
# https://arxiv.org/abs/1511.00363
#
# Was written by Itay Boneh and Asher Kabakovich, Tel-Aviv University
#

"""Tests the Binary-Connect network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import datetime
from utils import fill_feed_dict
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

import cifar10
import binary_connect as bc

# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 225000, 'Max number of steps.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size.  Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('log_dir', './log', 'Directory to put the log data.')
tf.app.flags.DEFINE_string('run_name', '', 'Name for the run (for logging).')
tf.app.flags.DEFINE_boolean('binary', True, 'Toggle binary-connect usage.')
tf.app.flags.DEFINE_boolean('stochastic', True, 'Switch between stochastic and deteministic binary-connect.')
tf.app.flags.DEFINE_string('model_path', './log/170219_200954binary_deterministicBIN_TrueSTOCH_True/model.ckpt', 'Path to a trained model.')


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
      batch_size: The batch size will be baked into both placeholders.

    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
      train_placeholder: Training mode indicator placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, cifar10.IM_ROWS, cifar10.IM_COLS, cifar10.IM_CH),
                                        name='images')
    labels_placeholder = tf.placeholder(tf.int32, shape=batch_size, name='labels')
    train_placeholder = tf.placeholder(tf.bool, name='is_train')
    return images_placeholder, labels_placeholder, train_placeholder


def run_testing():
    """Test BinaryConnect."""
    # Get the sets of images and labels for training, validation, and
    # test on CIFAR10.
    data_sets = cifar10.read_data_sets(dst_dir='./dataset', validation_size=5000)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder, train_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = bc.inference_bin(images_placeholder, train_placeholder,
                                  stochastic=FLAGS.stochastic,
                                  use_bnorm=True) \
            if FLAGS.binary \
            else bc.inference_ref(images_placeholder, train_placeholder,
                                  use_bnorm=True)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_metric = bc.evaluation(logits, labels_placeholder)

        # Add the variable initializer Op.
        ivars = tf.global_variables() + tf.local_variables()
        init = tf.variables_initializer(ivars)

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Load trained model.
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.model_path)
        print("Model loaded.")

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop
        duration = 0
        tp_value_total = 0
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular testing step.
            feed_dict = fill_feed_dict(data_sets.test,
                                       images_placeholder,
                                       labels_placeholder,
                                       train_placeholder, False)

            # Run one step of the model.
            acc_val = sess.run(eval_metric, feed_dict=feed_dict)
            duration += time.time() - start_time
            tp_value_total += acc_val

            # Print an overview
            if step % 100 == 0:
                # Print status to stdout.
                images_freq = 100 * FLAGS.batch_size / duration
                print('Step %d: correct = %.2f%% (%.3f images/sec)' %
                      (step, tp_value_total / step,
                       images_freq))
                duration = time.time() - start_time
                duration = 0

def main(_):
    FLAGS.run_name = \
        datetime.datetime.now().strftime("%y%m%d_%H%M%S") + \
        FLAGS.run_name + \
        'BIN_' + str(FLAGS.binary) + \
        'STOCH_' + str(FLAGS.stochastic)

    FLAGS.log_dir = os.path.join(FLAGS.log_dir, FLAGS.run_name)
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_testing()


if __name__ == '__main__':
    tf.app.run()
