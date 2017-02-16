import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.layers.convolutional import _Conv
import cifar10

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


def trainable_var(name, shape, initializer=tf.truncated_normal_initializer(stddev=.05), binary=FLAGS.binary):
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


def inference_ref2(input, is_train, use_bnorm=False):
    with tf.name_scope('128C3-128C3-P2'):
        x = tf.layers.conv2d(inputs=input, filters=128, kernel_size=3, padding="same", activation=tf.nn.relu,
                             use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=3, padding="same", activation=tf.nn.relu,
                             use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    with tf.name_scope('256C3-256C3-P2'):
        x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=3, padding="same", activation=tf.nn.relu,
                             use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=3, padding="same", activation=tf.nn.relu,
                             use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    with tf.name_scope('512C3-512C3-P2'):
        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=3, padding="same", activation=tf.nn.relu,
                             use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=3, padding="same", activation=tf.nn.relu,
                             use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    with tf.name_scope('1024FC-1024FC-10FC'):
        x = tf.reshape(x, [x.get_shape()[0].value, -1])
        x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu, use_bias=not use_bnorm)
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu, use_bias=not use_bnorm)
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.dense(inputs=x, units=cifar10.NB_CLASSES)

    return x


def inference_ref(input, is_train, use_bnorm=False):
    with tf.name_scope('conv1'):
        x = tf.layers.conv2d(inputs=input, filters=32, kernel_size=3, padding="same", activation=tf.nn.relu,
                             use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, padding="same", activation=tf.nn.relu,
                             use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    with tf.name_scope('conv2'):
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=3, padding="same", activation=tf.nn.relu,
                             use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=3, padding="same", activation=tf.nn.relu,
                             use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    with tf.name_scope('dense1'):
        # x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:])])
        x = tf.reshape(x, [x.get_shape()[0].value, -1])
        x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu)

    with tf.name_scope('dense2'):
        x = tf.layers.dense(inputs=x, units=cifar10.NB_CLASSES)

    return x


def inference(input, is_train):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      input: Images placeholder, from inputs().
      is_train: Training mode indicator placeholder.

    Returns:
      output_tensor: Output tensor with the computed logits.
    """

    # helper functions
    def conv2d_bnorm(input_tensor, kernel_size, input_dim, output_dim, layer_name, act=True, pool=False):
        """Reusable code for making a conv-bnorm-act neural net block.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name) as layer_scope:
            # This Variable will hold the state of the weights for the layer
            with tf.variable_scope('weights') as var_scope:
                # kernel = tf.cond(is_train,
                #                  lambda: trainable_var(layer_scope + '_' + var_scope.name,
                #                                        kernel_size + [input_dim, output_dim]),
                #                  lambda: trainable_var(layer_scope + '_' + var_scope.name,
                #                                        kernel_size + [input_dim, output_dim], binary=False))
                kernel = trainable_var(layer_scope + '_' + var_scope.name,
                                       kernel_size + [input_dim, output_dim], binary=False)
                variable_summaries(kernel)

            output_tensor = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')

            if pool:
                with tf.name_scope('pooling'):
                    output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME', name=layer_scope + 'P2')

            with tf.name_scope('batch_norm'):
                # output_tensor = _batch_norm(output_tensor, output_dim, is_train)
                output_tensor = batch_norm(output_tensor, is_training=is_train)

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
                # weights = tf.cond(is_train,
                #                   lambda: trainable_var(layer_scope + '_' + var_scope.name,
                #                                         [input_dim, output_dim], binary=True),
                #                   lambda: trainable_var(layer_scope + '_' + var_scope.name,
                #                                         [input_dim, output_dim], binary=False))
                weights = trainable_var(layer_scope + '_' + var_scope.name,
                                        [input_dim, output_dim], binary=False)
                variable_summaries(weights)

            output_tensor = tf.matmul(input_tensor, weights)

            with tf.name_scope('batch_norm'):
                # output_tensor = _batch_norm(output_tensor, output_dim, is_train)
                output_tensor = batch_norm(output_tensor, is_training=is_train)

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
    flat_sz = np.prod(x.get_shape().as_list()[1:])
    x = tf.reshape(x, [-1, flat_sz], name='reshape')
    x = dense_bnorm(x, flat_sz, 1024, '1024FC_1')
    x = dense_bnorm(x, 1024, 1024, '1024FC_2')
    x = dense_bnorm(x, 1024, 10, '10FC', act=False)

    return x


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)

    # if batch_norm is used, add dependency
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        with tf.control_dependencies([updates]):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='xentropy')
            # labels_oh = tf.one_hot(labels, cifar10.NB_CLASSES)
            # cross_entropy = tf.losses.hinge_loss(logits=logits, labels=labels_oh)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xentropy')
        # labels_oh = tf.one_hot(labels, cifar10.NB_CLASSES)
        # cross_entropy = tf.losses.hinge_loss(logits=logits, labels=labels_oh)
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    tp = tf.reduce_sum(tf.cast(correct, tf.int32))
    tf.summary.scalar('true_positive', tp)
    return tp


class Conv2D_bin(_Conv):
    """2D convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
      filters: integer, the dimensionality of the output space (i.e. the number
        output of filters in the convolution).
      kernel_size: an integer or tuple/list of 2 integers, specifying the
        width and height of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: an integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, width, height, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, width, height)`.
      dilation_rate: an integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, no bias will
        be applied.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Regularizer function for the output.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      name: A string, the name of the layer.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        self.stochastic = kwargs.get('stochastic', False)
        if 'stochastic' in kwargs.keys():
            del kwargs['stochastic']

        super(Conv2D_bin, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            name=name, **kwargs)

    def build(self, input_shape):
        tmp = super(Conv2D_bin).build(input_shape)
        self.kernel_b = tf.get_variable('kernel',
                                        shape=self.kernel.get_shape(),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        trainable=True,
                                        dtype=self.dtype)
        self.kernel.trainble = False
        return tmp

    def call(self, inputs):
        return super(Conv2D_bin).call(inputs)


def conv2d_bin(inputs,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               reuse=None):
    """Functional interface for the 2D convolution layer.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
      inputs: Tensor input.
      filters: integer, the dimensionality of the output space (i.e. the number
        output of filters in the convolution).
      kernel_size: an integer or tuple/list of 2 integers, specifying the
        width and height of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: an integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, width, height, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, width, height)`.
      dilation_rate: an integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, no bias will
        be applied.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Regularizer function for the output.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      name: A string, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      Output tensor.
    """
    layer = Conv2D_bin(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs)
