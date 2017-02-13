import cifar10
import utils
import binary_connect as bc
import tensorflow as tf
from tensorflow.contrib.losses import hinge_loss

FLAGS = tf.app.flags.FALGS

tf.app.flags.DEFINE_integer('max_steps', 1e6, 'Max number of epochs')
tf.app.flags.DEFINE_string('log_dir', './log', 'Folder Tensorboard logs')

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# Get the data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = utils.to_categorical(y_train, nb_classes)
Y_test = utils.to_categorical(y_test, nb_classes)

# Input placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    is_train = tf.placeholder(tf.bool, name='is_train')

with tf.name_scope('prediction'):
    out = bc.model(x, is_train)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope('loss'):
    loss = tf.reduce_mean(hinge_loss(out, labels=y))

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer().minimize(loss)

# TODO: split data to batches
def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
        xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
    else:
        xs, ys = mnist.test.images, mnist.test.labels
    return {x: xs, y: ys, is_train: train}


# Train the model, and also write summaries.
# Every 10th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data and add training summaries

with tf.Session() as sess:
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    # TODO: run training always. run validation every _ steps
    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
