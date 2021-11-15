# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class Model(object):
  """ResNet model."""

  def __init__(self, restore=None, sess=None, tiny=True, mode='eval'):
    """ResNet constructor.

    Args:
      mode: One of 'train' and 'eval'.
    """
    
    tf.compat.v1.disable_eager_execution()
    
    self.mode = mode
    self.tiny = tiny
    with tf.compat.v1.variable_scope('model', reuse=tf.compat.v1.AUTO_REUSE):
      self._build_model()
    if restore:
      path = tf.compat.v1.train.latest_checkpoint(restore)
      used_vars = [x for x in tf.compat.v1.global_variables() if x.name.startswith('model')]
      #other_dict = {x.name[6:-2]:x for x in tf.compat.v1.global_variables() if x.name.startswith('model')}
      saver = tf.compat.v1.train.Saver(var_list=used_vars)
      saver.restore(sess, path)
      
  def __call__(self, xs):
    with tf.compat.v1.variable_scope('model', reuse=tf.compat.v1.AUTO_REUSE):
      return self._build_model(xs)
    
  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.compat.v1.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self, x_input = None):
    assert self.mode == 'train' or self.mode == 'eval'
    """Build the core model within the graph."""
    with tf.compat.v1.variable_scope('input'):

      if x_input == None:
        self.x_input = tf.compat.v1.placeholder(
          tf.compat.v1.float32,
          shape=[None, 32, 32, 3], name='x_input_model')
      else:
        self.x_input = x_input

      self.y_input = tf.compat.v1.placeholder(tf.compat.v1.int64, shape=None, name='y_input_model')


      input_standardized = tf.compat.v1.map_fn(lambda img: tf.compat.v1.image.per_image_standardization(img),
                               self.x_input)
      x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))



    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual

    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    if self.tiny:
      filters = [16, 16, 32, 64]
      layers = 2
    else:
      filters = [16, 160, 320, 640]
      layers = 5

    # Update hps.num_residual_units to 9

    with tf.compat.v1.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, layers):
      with tf.compat.v1.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.compat.v1.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, layers):
      with tf.compat.v1.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.compat.v1.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, layers):
      with tf.compat.v1.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.compat.v1.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.1)
      x = self._global_avg_pool(x)

    with tf.compat.v1.variable_scope('logit'):
      self.pre_softmax = self._fully_connected(x, 10)

    self.predictions = tf.compat.v1.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.compat.v1.equal(self.predictions, self.y_input)
    self.num_correct = tf.compat.v1.reduce_sum(
        tf.compat.v1.cast(self.correct_prediction, tf.compat.v1.int64))
    self.accuracy = tf.compat.v1.reduce_mean(
        tf.compat.v1.cast(self.correct_prediction, tf.compat.v1.float32))

    with tf.compat.v1.variable_scope('costs'):
      self.y_xent = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.pre_softmax, labels=self.y_input)
      self.xent = tf.compat.v1.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.compat.v1.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()
    return self.pre_softmax

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.compat.v1.name_scope(name):
      # return tf.compat.v1.contrib.layers.batch_norm(
      return tf.contrib.batch_normalization(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=(self.mode == 'train'))

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.compat.v1.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.compat.v1.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.compat.v1.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.compat.v1.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.compat.v1.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.compat.v1.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.compat.v1.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.compat.v1.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.compat.v1.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.compat.v1.nn.l2_loss(var))
    return tf.compat.v1.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.compat.v1.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.compat.v1.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.compat.v1.float32, initializer=tf.compat.v1.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.compat.v1.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.compat.v1.where(tf.compat.v1.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.compat.v1.reshape(x, [tf.compat.v1.shape(x)[0], -1])
    w = tf.compat.v1.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.compat.v1.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.compat.v1.get_variable('biases', [out_dim],
                        initializer=tf.compat.v1.constant_initializer())
    return tf.compat.v1.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.compat.v1.reduce_mean(x, [1, 2])



