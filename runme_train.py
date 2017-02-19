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

"""Trains and Evaluates the Binary-Connect network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import datetime
from utils import do_eval, fill_feed_dict
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

import cifar10
import binary_connect as bc

# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 225000, 'Max number of steps.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size.  Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_integer('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_string('log_dir', '.\\log', 'Directory to put the log data.')
tf.app.flags.DEFINE_string('run_name', '', 'Name for the run (for logging).')
tf.app.flags.DEFINE_boolean('binary', True, 'Toggle binary-connect usage.')
tf.app.flags.DEFINE_boolean('stochastic', True, 'Switch between stochastic and deteministic binary-connect.')


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


def run_training():
    """Train BinaryConnect."""
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

        # Add to the Graph the Ops for loss calculation.
        loss = bc.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = bc.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_metric = bc.evaluation(logits, labels_placeholder)

        # Add a placeholder for logging execution time
        # frequency_placeholder = tf.placeholder(tf.float32, shape=())
        # tf.summary.scalar('Execution Time', frequency_placeholder)
        # TODO: support a d separate summary for metadata (e.g. execution time)


        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        ivars = tf.global_variables() + tf.local_variables()
        init = tf.variables_initializer(ivars)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a logger to the validation accuracy
        val_acc_pl = tf.placeholder(tf.float32, shape=())
        summary_val_acc = tf.summary.scalar(name='validation_acc', tensor=val_acc_pl, collections=['validation'])
        summary_val = tf.summary.merge([summary_val_acc])

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer_train = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'), sess.graph)
        summary_writer_val = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'val'), sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        duration = 0
        tp_value_total = 0
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder,
                                       train_placeholder, True)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value, acc_val = sess.run([train_op, loss, eval_metric],
                                              feed_dict=feed_dict)

            duration += time.time() - start_time
            tp_value_total += acc_val

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0 and step > 0:
                # Print status to stdout.
                images_freq = 100 * FLAGS.batch_size / duration
                print('Step %d: loss = %.2f, correct = %.2f%% (%.3f images/sec)' %
                      (step, loss_value, tp_value_total / FLAGS.batch_size,
                       images_freq))
                duration = time.time() - start_time
                tp_value_total = 0
                duration = 0
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer_train.add_summary(summary_str, step)
                summary_writer_train.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 500 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                # print('Training Data Eval:')
                # do_eval(sess,
                #         eval_metric,
                #         images_placeholder,
                #         labels_placeholder,
                #         train_placeholder,
                #         data_sets.train, summary)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                accuracy_val = do_eval(sess, eval_metric, images_placeholder, labels_placeholder, train_placeholder,
                                       data_sets.validation)
                # TODO: find a way to collect summaries for validation
                summary_str = sess.run(summary_val, feed_dict={val_acc_pl: accuracy_val})
                summary_writer_val.add_summary(summary_str, step)
                summary_writer_val.flush()

                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess, eval_metric, images_placeholder, labels_placeholder, train_placeholder, data_sets.test)


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
    run_training()


if __name__ == '__main__':
    tf.app.run()
