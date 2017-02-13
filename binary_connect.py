import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('binary', False, "Enable binary connect")
tf.app.flags.DEFINE_boolean('stochastic', False, "Use stochastic binarization")

INPUT_SIZE = (32, 32, 3)


def hard_sig(x):
    x = tf.clip_by_value((x + 1.) / 2., 0, 1)
    return x


def binarize_deterministic(w):
    w = hard_sig(w)
    wb = tf.round(w)
    wb = tf.select(tf.equal(wb, 1.), tf.ones_like(wb), -tf.ones_like(wb))
    return wb


def binarize_stochastic(w):
    w = hard_sig(w)
    wb = tf.to_float(tf.random_uniform(tf.shape(w), 0, 1.) <= w)
    wb = tf.select(tf.equal(wb, 1.), tf.ones_like(wb), -tf.ones_like(wb))
    return wb


def trainable_var(name, shape, initializer=tf.truncated_normal_initializer(), binary=FLAGS.binary):
    """Helper to create an initialized Variable with or without binarization.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for the variable
      binary: use binary connect or not
      stochastic: use stochastic binarization

    Returns:
      Variable Tensor
    """
    if binary:
        # create a variable and prevent it from being added to the TRAINABLE_VARIABLES collection
        var_b = tf.get_variable(name + "_b", shape,
                                initializer=initializer,
                                trainable=False)

        var_t = tf.get_variable(name + "_t", shape,
                                initializer=initializer,
                                trainable=False)

        with tf.control_dependencies([var_t]):
            if FLAGS.stochastic:
                var_b.assign(binarize_stochastic(var_t))
            else:
                var_b.assign(binarize_deterministic(var_t))

        # add the
        tf.add_to_collection("BINARY_TRAINABLE", (var_b, var_t))
    else:
        var_b = tf.get_variable(name, shape,
                                initializer=initializer,
                                trainable=True)

    return var_b


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def _batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def model(input, is_train):
    # helper functions
    def conv2d_bnorm(input_tensor, kernel_size, input_dim, output_dim, layer_name, act=True, pool=False):
        """Reusable code for making a conv-bnorm-act neural net block.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name) as layer_scope:
            # This Variable will hold the state of the weights for the layer
            with tf.variable_scope('weights') as var_scope:
                kernel = tf.cond(is_train,
                                 lambda: trainable_var(layer_scope.name + '_' + var_scope.name,
                                                       kernel_size + [input_dim, output_dim], binary=True),
                                 lambda: trainable_var(layer_scope.name + '_' + var_scope.name,
                                                       kernel_size + [input_dim, output_dim], binary=False))
                variable_summaries(kernel)

            output_tensor = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')

            if pool:
                with tf.name_scope('pooling'):
                    output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME', name=layer_scope.name + 'P2')

            with tf.name_scope('batch_norm'):
                output_tensor = _batch_norm(output_tensor, output_dim, is_train)

            if act:
                with tf.name_scope('activation'):
                    output_tensor = tf.nn.relu(output_tensor, name='activation')
                    tf.summary.histogram('activations', output_tensor)

            return output_tensor

    def dense_bnorm(input_tensor, input_dim, output_dim, layer_name, act=True):
        """Reusable code for making a dense-bnorm-act neural net block.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name) as layer_scope:
            # This Variable will hold the state of the weights for the layer
            with tf.variable_scope('weights') as var_scope:
                weights = tf.cond(is_train,
                                  lambda: trainable_var(layer_scope.name + '_' + var_scope.name,
                                                        [input_dim, output_dim], binary=True),
                                  lambda: trainable_var(layer_scope.name + '_' + var_scope.name,
                                                        [input_dim, output_dim], binary=False))
                variable_summaries(weights)

            output_tensor = tf.matmul(input_tensor, weights)

            with tf.name_scope('batch_norm'):
                output_tensor = _batch_norm(output_tensor, output_dim, is_train)

            if act:
                with tf.name_scope('activation'):
                    output_tensor = tf.nn.relu(output_tensor, name='activation')
                    tf.summary.histogram('activations', output_tensor)

            return output_tensor

    # build model

    # 128C3-128C3-P2
    x = conv2d_bnorm(input, [3, 3], INPUT_SIZE[-1], 128, '128C3_1')
    x = conv2d_bnorm(x, [3, 3], 128, 128, '128C3_2', pool=True)

    # 256C3-256C3-P2
    x = conv2d_bnorm(x, [3, 3], 128, 256, '256C3_1')
    x = conv2d_bnorm(x, [3, 3], 256, 256, '256C3_2', pool=True)

    # 512C3-512C3-P2
    x = conv2d_bnorm(x, [3, 3], 256, 512, '512C3_1')
    x = conv2d_bnorm(x, [3, 3], 512, 512, '512C3_2', pool=True)

    # 1024FC-1024FC-10FC
    x = dense_bnorm(x, 512, 1024, '1024FC_1')
    x = dense_bnorm(x, 1024, 1024, '1024FC_2')
    x = dense_bnorm(x, 1024, 10, '10FC', act=False)

    return x
