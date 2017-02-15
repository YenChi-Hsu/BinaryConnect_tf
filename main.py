from cifar10 import read_data_sets
import binary_connect as bc
import tensorflow as tf
from tensorflow.contrib.losses import hinge_loss
import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 50, 'Max number of epochs')
tf.app.flags.DEFINE_string('log_dir', './log', 'Folder Tensorboard logs')

learning_rate = 0.01
batch_size = 64
display_step = 20
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3
dtstr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")


def main(argv=None):
    # Get the data, shuffled and split between train and test sets:
    cifar10 = read_data_sets(dst_dir='./dataset', validation_size=5000)
    print(cifar10.train.num_examples, 'train samples')
    print(cifar10.validation.num_examples, 'validation samples')
    print(cifar10.test.num_examples, 'test samples')

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')
        train = tf.placeholder(tf.bool, name='train')

    with tf.name_scope('prediction'):
        out = bc.model(x, train)

    with tf.name_scope('accuracy') as nm_scope:
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar(nm_scope, accuracy)

    with tf.name_scope('loss') as nm_scope:
        loss = tf.reduce_mean(hinge_loss(out, labels=y))
        tf.summary.scalar(nm_scope, loss)

    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer().minimize(loss)

    # with tf.name_scope('Average_summary'):
    #     avg_loss_ph = tf.placeholder(tf.float32)
    #     avg_acc_ph = tf.placeholder(tf.float32)
    #     average_loss = tf.scalar_summary("average_loss", avg_loss_ph)
    #     average_acc = tf.scalar_summary("average_acc", avg_acc_ph)

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data and add training summaries
    # Launch the graph
    with tf.Session() as sess:
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train/' + dtstr, sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test/' + dtstr)
        tf.global_variables_initializer().run()

        # Keep training until reach max iterations
        global_step = 0
        for epoch in range(FLAGS.max_steps):
            # train epoch
            step = 0
            while step * batch_size < cifar10.train.num_examples:
                global_step += 1
                batch_x, batch_y = cifar10.train.next_batch(batch_size)
                # Run optimization op
                summary, _ = sess.run([merged, train_op], feed_dict={x: batch_x,
                                                                     y: batch_y,
                                                                     train: True})
                # train_writer.add_summary(summary, train_step)
                if step > 0 and step % display_step == 0:
                    # Calculate batch loss, accuracy and evaluation time
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, loss_v, acc_v = sess.run([merged, loss, accuracy], options=run_options,
                                                      run_metadata=run_metadata, feed_dict={x: batch_x,
                                                                                            y: batch_y,
                                                                                            train: False})
                    train_writer.add_summary(summary, global_step)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % global_step)
                    print(
                        "Epoch {:d}, Step {:d} ({:d}/{:d} samples), Minibatch Loss={:.6f}, Training Accuracy={:.5f}".format(
                            epoch, step, step * batch_size,
                            cifar10.train.num_examples, loss_v, acc_v))
                    # print("Epoch " + str(epoch) + ", Step " + str(step) + ", Minibatch Loss= " + \
                    #       "{:.6f}".format(loss_v) + ", Training Accuracy= " + \
                    #       "{:.5f}".format(acc_v))
                step += 1

            # TODO: at the end of every epoch, preform a validation epoch and update tensorboard summaries
            # # validation epoch
            # step = 0
            # losses = []
            # accs = []
            # while step * batch_size < cifar10.validation.num_examples:
            #     batch_x, batch_y = cifar10.validation.next_batch(batch_size)
            #     summary, loss_v, acc_v = sess.run([merged, loss, accuracy],
            #                                       feed_dict={x: batch_x,
            #                                                  y: batch_y,
            #                                                  train: False})
            #     losses.append(loss_v)
            #     accs.append(acc_v)
            #     print(
            #         "Epoch {:d}, Step {:d} ({:d}/{:d} samples), Minibatch Loss={:.6f}, Validation Accuracy={:.5f}".format(
            #             epoch, step, step * batch_size,
            #             cifar10.validation.num_examples, loss_v, acc_v))
            #
            # sum_loss, sum_acc = sess.run([average_loss, average_acc],
            #                              feed_dict={
            #                                  avg_loss_ph: sum(losses) / len(losses),
            #                                  avg_acc_ph: sum(accs) / len(accs)
            #                              })
            # train_writer.add_summary(sum_loss, global_step)
            # train_writer.add_summary(sum_acc, global_step)

        print("Optimization Finished!")
        train_writer.close()
        validation_writer.close()

        # step = 0
        # while step * batch_size < cifar10.validation.num_examples:
        #     # Calculate batch loss, accuracy and evaluation time
        #     batch_x, batch_y = cifar10.validation.next_batch(batch_size)
        #     summary, _, acc_v = sess.run([merged, loss, accuracy],
        #                                  feed_dict={x: batch_x,
        #                                             y: batch_y,
        #                                             train: False})
        #     print("Epoch " + str(epoch) +
        #           ", Step " + str(step) +
        #           ", Validation Accuracy= " + \
        #           "{:.5f}".format(acc_v))
        #     step += 1

        # Calculate accuracy for 256 mnist test images
        acc = sess.run(accuracy, feed_dict={x: cifar10.test.images,
                                            y: cifar10.test.labels,
                                            train: False})
        print("Testing Accuracy:" + str(acc))


if __name__ == '__main__':
    tf.app.run()
