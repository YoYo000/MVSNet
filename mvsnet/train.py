#!/usr/bin/env python
"""
Copyright 2018, Yao Yao, HKUST.
Training script.
"""

from __future__ import print_function

import os
import time
import sys
import math
import argparse
from random import randint

import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

sys.path.append("../")
from tools.common import Notify

from preprocess import *
from model import inference, mvsnet_loss, depth_refine
from homography_warping import get_homographies, homography_warping

# params for datasets
tf.app.flags.DEFINE_string('dtu_data_root', '/data/dtu/', 
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('log_dir', '/data/tf_log',
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('save_dir', '/data/tf_model',
                           """Path to save the model.""")
tf.app.flags.DEFINE_boolean('train_dtu', True, 
                            """Whether to train.""")

# params for training
tf.app.flags.DEFINE_integer('num_gpus', 1, 
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('view_num', 3, 
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128, 
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 640, 
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 512, 
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25, 
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_integer('base_image_size', 128, 
                            """Base image size to fit the network.""")
tf.app.flags.DEFINE_float('interval_scale', 1.6, 
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """training batch size""")
tf.app.flags.DEFINE_integer('epoch', 6, 
                            """training epoch""")
tf.app.flags.DEFINE_float('val_ratio', 0, 
                          """ratio of validation set when splitting dataset.""")

# params for config
tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', None,
                           """Path to restore the model.""")
# tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', 
#                            '/home/yoyo/Documents/software/tfmatch/mvs/tf_model/model.ckpt',
#                            """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 0,
                            """ckpt step.""")
tf.app.flags.DEFINE_boolean('is_training', True,
                            """Flag to training model""")

# Params for solver.
tf.app.flags.DEFINE_float('base_lr', 0.001,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 10000,
                            """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 5000,
                            """Step interval to save the model.""")
tf.app.flags.DEFINE_float('gamma', 0.9,
                          """Learning rate decay rate.""")

FLAGS = tf.app.flags.FLAGS

class MVSGenerator:
    """ data generator class, tf only accept generator without param """
    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
        self.view_num = view_num
        self.sample_num = len(sample_list)
        self.counter = 0
    
    def __iter__(self):
        while True:
            for data in self.sample_list: 
                start_time = time.time()

                ###### read input data ######
                images = []
                cams = []
                for view in range(self.view_num):
                    image = center_image(cv2.imread(data[2 * view]))
                    cam = load_cam(open(data[2 * view + 1]))
                    cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale
                    images.append(image)
                    cams.append(cam)
                depth_image = load_pfm(open(data[2 * self.view_num]))

                # mask out-of-range depth pixels (in a relaxed range)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                depth_image = mask_depth_image(depth_image, depth_start, depth_end)

                # return mvs input
                self.counter += 1
                duration = time.time() - start_time
                # print(Notify.INFO, 'prepocess: %.3f sec/step)' % duration, Notify.ENDC)
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                yield (images, cams, depth_image) 

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(traning_list):
    """ training mvsnet """
    training_sample_size = len(traning_list)
    print ('sample number: ', training_sample_size)

    with tf.Graph().as_default(), tf.device('/cpu:0'): 
        ########## data iterator #########
        # training generators
        training_generator = iter(MVSGenerator(traning_list, FLAGS.view_num))
        generator_data_type = (tf.float32, tf.float32, tf.float32)
        # dataset from generator
        training_set = tf.data.Dataset.from_generator(lambda: training_generator, generator_data_type)
        training_set = training_set.batch(FLAGS.batch_size)
        training_set = training_set.prefetch(buffer_size=1)
        # iterators
        training_iterator = training_set.make_initializable_iterator()

        ########## optimization options ##########
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr_op = tf.train.exponential_decay(FLAGS.base_lr, global_step=global_step, 
                                           decay_steps=FLAGS.stepvalue, decay_rate=FLAGS.gamma, name='lr')
        opt = tf.train.RMSPropOptimizer(learning_rate=lr_op)

        tower_grads = []
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    # generate data
                    images, cams, depth_image = training_iterator.get_next()
                    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
                    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
                    depth_image.set_shape(tf.TensorShape([None, None, None, 1]))
                    depth_start = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])

                    is_master_gpu = False
                    if i == 0:
                        is_master_gpu = True

                    # inference
                    depth_map, prob_map = inference(
                        images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                    # refinement 
                    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                    refined_depth_map = depth_refine(
                        depth_map, ref_image, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                    
                    # loss
                    loss0, less_one_temp, less_three_temp = mvsnet_loss(
                        depth_map, depth_image, depth_interval)
                    loss1, less_one_accuracy, less_three_accuracy = mvsnet_loss(
                        refined_depth_map, depth_image, depth_interval)
                    loss = (loss0 + loss1) / 2

                    # retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # keep track of the gradients across all towers.
                    tower_grads.append(grads)
        
        # average gradient
        grads = average_gradients(tower_grads)
        
        # training opt
        train_opt = opt.apply_gradients(grads, global_step=global_step)

        # summary 
        summaries.append(tf.summary.scalar('loss', loss))
        summaries.append(tf.summary.scalar('less_one_accuracy', less_one_accuracy))
        summaries.append(tf.summary.scalar('less_three_accuracy', less_three_accuracy))
        summaries.append(tf.summary.scalar('lr', lr_op))
        weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in weights_list:
            summaries.append(tf.summary.histogram(var.op.name, var))
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        
        # saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)        
        summary_op = tf.summary.merge(summaries)

        # initialization option
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:     
            
            # initialization
            total_step = 0
            sess.run(init_op)
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

            # load pre-trained model
            if FLAGS.pretrained_model_ckpt_path is not None:
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(
                    sess, '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                    ('-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
                total_step = FLAGS.ckpt_step

            # training several epochs
            for epoch in range(FLAGS.epoch):

                # training of one epoch
                step = 0
                sess.run(training_iterator.initializer)
                for _ in range(int(training_sample_size / FLAGS.num_gpus)):

                    # run one batch
                    start_time = time.time()
                    try:
                        out_summary_op, out_opt, out_loss, out_less_one, out_less_three = sess.run(
                        [summary_op, train_opt, loss, less_one_accuracy, less_three_accuracy])
                    except tf.errors.OutOfRangeError:
                        print("End of dataset")  # ==> "End of dataset"
                        break
                    duration = time.time() - start_time

                    # print info
                    if step % FLAGS.display == 0:
                        print(Notify.INFO,
                            'epoch, %d, step %d, total_step %d, loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                            (epoch, step, total_step, out_loss, out_less_one, out_less_three, duration), Notify.ENDC)
                    
                    # write summary
                    if step % (FLAGS.display * 10) == 0 and FLAGS.is_training:
                        summary_writer.add_summary(out_summary_op, total_step)
                   
                    # save the model checkpoint periodically
                    if (total_step % FLAGS.snapshot == 0 or step == (training_sample_size - 1)) and FLAGS.is_training:
                        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=total_step)
                    step += FLAGS.batch_size * FLAGS.num_gpus
                    total_step += FLAGS.batch_size * FLAGS.num_gpus

def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # Prepare all training samples
    sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root)
    # Shuffle
    random.shuffle(sample_list)
    # Training entrance.
    train(sample_list)


if __name__ == '__main__':
    print ('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()