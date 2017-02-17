import numpy as np
import tensorflow as tf
from tensorflow.python import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.layers.convolutional import _Conv
from tensorflow.python.layers.core import Dense
import cifar10

FLAGS = tf.app.flags.FLAGS

INPUT_SIZE = (32, 32, 3)


def hard_sig(x):
    x = tf.clip_by_value((x + 1.) / 2., 0, 1)
    return x


def binarize(w, stochastic=False):
    if stochastic:
        with tf.name_scope('binarize_stochastic'):
            w = hard_sig(w)
            wb = tf.to_float(tf.random_uniform(tf.shape(w), 0, 1.) <= w)
            wb = tf.where(tf.equal(wb, 1.), tf.ones_like(wb), -tf.ones_like(wb))
    else:
        with tf.name_scope('binarize_deterministic'):
            wb = tf.where(tf.greater_equal(w, 0.), tf.ones_like(w), -tf.ones_like(w))
    return wb


def variable_summaries(var, name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries_' + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def inference_bin(input, is_train, stochastic=False, use_bnorm=False):
    with tf.name_scope('128C3-128C3-P2'):
        x = conv2d_bin(stochastic=stochastic, inputs=input, filters=128, kernel_size=3, padding="same",
                       activation=tf.nn.relu,
                       use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = conv2d_bin(stochastic=stochastic, inputs=x, filters=128, kernel_size=3, padding="same",
                       activation=tf.nn.relu,
                       use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    with tf.name_scope('256C3-256C3-P2'):
        x = conv2d_bin(stochastic=stochastic, inputs=x, filters=256, kernel_size=3, padding="same",
                       activation=tf.nn.relu,
                       use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = conv2d_bin(stochastic=stochastic, inputs=x, filters=256, kernel_size=3, padding="same",
                       activation=tf.nn.relu,
                       use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    with tf.name_scope('512C3-512C3-P2'):
        x = conv2d_bin(stochastic=stochastic, inputs=x, filters=512, kernel_size=3, padding="same",
                       activation=tf.nn.relu,
                       use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = conv2d_bin(stochastic=stochastic, inputs=x, filters=512, kernel_size=3, padding="same",
                       activation=tf.nn.relu,
                       use_bias=not use_bnorm, kernel_initializer=init_ops.glorot_normal_initializer())
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    with tf.name_scope('1024FC-1024FC-10FC'):
        fun = tf.layers.dense if False else dense_bin
        x = tf.reshape(x, [x.get_shape()[0].value, -1])
        x = fun(inputs=x, units=1024, activation=tf.nn.relu, use_bias=not use_bnorm)
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = fun(inputs=x, units=1024, activation=tf.nn.relu, use_bias=not use_bnorm)
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = fun(inputs=x, units=cifar10.NB_CLASSES)

    return x


def inference_ref(input, is_train, use_bnorm=False):
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
        fun = tf.layers.dense if True else dense_bin
        x = tf.reshape(x, [x.get_shape()[0].value, -1])
        x = fun(inputs=x, units=1024, activation=tf.nn.relu, use_bias=not use_bnorm)
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = fun(inputs=x, units=1024, activation=tf.nn.relu, use_bias=not use_bnorm)
        if use_bnorm:
            x = tf.layers.batch_normalization(inputs=x, training=is_train)
        x = fun(inputs=x, units=cifar10.NB_CLASSES)

    return x


def inference_ref2(input, is_train, use_bnorm=False):
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
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     logits=logits, labels=labels, name='xentropy')
            labels_oh = tf.one_hot(labels, cifar10.NB_CLASSES)
            cross_entropy = tf.square(tf.losses.hinge_loss(logits=logits, labels=labels_oh))
            loss_t = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    else:
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=logits, labels=labels, name='xentropy')
        labels_oh = tf.one_hot(labels, cifar10.NB_CLASSES)
        cross_entropy = tf.losses.hinge_loss(logits=logits, labels=labels_oh)
        loss_t = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss_t


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
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # # Use the optimizer to apply the gradients that minimize the loss
    # # (and also increment the global step counter) as a single training step.
    # train_op = optimizer.minimize(loss, global_step=global_step)
    # Use the optimizer to compute gradients
    grads_and_vars = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    # Replace binary variables with their continuous pairs
    grads_and_vars_for_update = [(g, v.grad_update_var) if hasattr(v, 'grad_update_var') else (g, v) for g, v in
                                 grads_and_vars]
    # Apply gradient updates
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars_for_update, global_step=global_step)

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
        tmp = super(Conv2D_bin, self).build(input_shape)
        self.kernel_t = tf.get_variable('kernel_t',
                                        shape=self.kernel.get_shape(),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        trainable=False,
                                        dtype=self.dtype)
        self.kernel.grad_update_var = self.kernel_t
        variable_summaries(self.kernel, 'kernel')
        variable_summaries(self.kernel_t, 'kernel_t')
        return tmp

    def call(self, inputs):
        assign_t = tf.assign(self.kernel, binarize(self.kernel_t, stochastic=self.stochastic))
        with tf.control_dependencies([assign_t]):
            return super(Conv2D_bin, self).call(inputs)


def conv2d_bin(inputs,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               stochastic=False,
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
        _scope=name,
        stochastic=stochastic)
    return layer.apply(inputs)


class Dense_bin(Dense):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, trainable=True, name=None, **kwargs):
        self.stochastic = kwargs.get('stochastic', False)
        if 'stochastic' in kwargs.keys():
            del kwargs['stochastic']

        super(Dense_bin, self).__init__(units, activation, use_bias, kernel_initializer, bias_initializer,
                                        kernel_regularizer,
                                        bias_regularizer, activity_regularizer, trainable, name, **kwargs)

    def build(self, input_shape):
        tmp = super(Dense_bin, self).build(input_shape)
        self.kernel_t = tf.get_variable('kernel_t',
                                        shape=self.kernel.get_shape(),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        trainable=False,
                                        dtype=self.dtype)
        self.kernel.grad_update_var = self.kernel_t
        variable_summaries(self.kernel, 'kernel')
        variable_summaries(self.kernel_t, 'kernel_t')
        return tmp

    def call(self, inputs):
        assign_t = tf.assign(self.kernel, binarize(self.kernel_t, stochastic=self.stochastic))
        # assign_t = tf.Print(assign_t, [self.kernel, self.kernel_t])
        with tf.control_dependencies([assign_t]):
            return super(Dense_bin, self).call(inputs)


def dense_bin(
        inputs, units,
        activation=None,
        use_bias=True,
        stochastic=False,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):
    layer = Dense_bin(units,
                      activation=activation,
                      use_bias=use_bias,
                      stochastic=stochastic,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer,
                      trainable=trainable,
                      name=name,
                      dtype=inputs.dtype.base_dtype,
                      _scope=name,
                      _reuse=reuse)
    return layer.apply(inputs)
