#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Convolutional GRU module.
"""

import tensorflow as tf

def group_norm(input_tensor,
               name,
               channel_wise=True,
               group=32,
               group_channel=8,
               is_reuse=False):

    x = tf.transpose(input_tensor, [0, 3, 1, 2])

    # shapes and groups
    shape = tf.shape(x)
    N = shape[0]
    C = x.get_shape()[1]
    H = shape[2]
    W = shape[3]
    if channel_wise:
        G = max(1, C / group_channel)
    else:
        G = min(group, C)

    # use layer normalization to simplify operations
    if G == 1:
        return tf.contrib.layers.layer_norm(input_tensor)

    # use instance normalization to simplify operations
    elif G >= C:
        return tf.contrib.layers.instance_norm(input_tensor)
    
    # group normalization as suggested in the paper
    else:
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + 1e-5)

        # per channel scale and bias (gamma and beta)
        with tf.variable_scope(name + '/gn', reuse=is_reuse):        
            gamma = tf.get_variable('gamma', [C], dtype=tf.float32, initializer=tf.ones_initializer())
            beta = tf.get_variable('beta', [C], dtype=tf.float32, initializer=tf.zeros_initializer())
            
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        output = tf.transpose(output, [0, 2, 3, 1])
        return output

class ConvGRUCell(tf.contrib.rnn.RNNCell):
    """A GRU cell with convolutions instead of multiplications."""

    def __init__(self,
                 shape,
                 filters,
                 kernel,
                 initializer=None,
                 activation=tf.tanh,
                 normalize=True):
        self._filters = filters
        self._kernel = kernel
        self._initializer = initializer
        self._activation = activation
        self._size = tf.TensorShape(shape + [self._filters])
        self._normalize = normalize
        self._feature_axis = self._size.ndims

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def __call__(self, x, h, scope=None):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            with tf.variable_scope('Gates'):
                
                # concatenation
                inputs = tf.concat([x, h], axis=self._feature_axis)

                # convolution
                conv = tf.layers.conv2d(
                    inputs, 2 * self._filters, self._kernel, padding='same', name='conv')
                reset_gate, update_gate = tf.split(conv, 2, axis=self._feature_axis)

                # group normalization, actually is 'instance normalization' as to save GPU memory 
                reset_gate = group_norm(reset_gate, 'reset_norm', group_channel=16)
                update_gate = group_norm(update_gate, 'update_norm', group_channel=16)

                # activation
                reset_gate = tf.sigmoid(reset_gate)
                update_gate = tf.sigmoid(update_gate)

            with tf.variable_scope('Output'):

                # concatenation
                inputs = tf.concat([x, reset_gate * h], axis=self._feature_axis)

                # convolution
                conv = tf.layers.conv2d(
                    inputs, self._filters, self._kernel, padding='same', name='output_conv')

                # group normalization
                conv = group_norm(conv, 'output_norm', group_channel=16)
                    
                # activation
                y = self._activation(conv)

                # soft update
                output = update_gate * h + (1 - update_gate) * y

            return output, output