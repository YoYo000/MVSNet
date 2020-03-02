#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import matplotlib.pyplot as plt

sys.path.append("../")
from tools.common import Notify

from preprocess import *
from model import *
from loss import * 
from homography_warping import get_homographies, homography_warping
import photometric_augmentation as photaug

# paths
tf.app.flags.DEFINE_string('blendedmvs_data_root', '/data/BlendedMVS/dataset_low_res', 
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('eth3d_data_root', '/data/eth3d/lowres/training/undistorted', 
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('dtu_data_root', '/data/dtu', 
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_boolean('train_blendedmvs', False, 
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('train_dtu', False, 
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('train_eth3d', False, 
                            """Whether to train.""")
tf.app.flags.DEFINE_string('log_folder', '/data/tf_log',
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_folder', '/data/tf_model',
                           """Path to save the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 0,
                            """ckpt step.""")
tf.app.flags.DEFINE_boolean('use_pretrain', False, 
                            """Whether to train.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 3, 
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 192, 
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 640, 
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 512, 
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25, 
                            """Downsample scale for building cost volume.""")

# network architectures
tf.app.flags.DEFINE_string('regularization', 'GRU',
                           """Regularization method.""")
tf.app.flags.DEFINE_boolean('refinement', False,
                           """Whether to apply depth map refinement for 3DCNNs""")

# training parameters
tf.app.flags.DEFINE_integer('num_gpus', 1, 
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', 6, 
                            """Training epoch number.""")
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
tf.app.flags.DEFINE_boolean('online_augmentation', False,
                           """Whether to apply image online augmentation during training""")

FLAGS = tf.app.flags.FLAGS


def online_augmentation(image, random_order=True):
    primitives = photaug.augmentations
    config = {}
    config['random_brightness'] = {'max_abs_change': 50}
    config['random_contrast'] = {'strength_range': [0.3, 1.5]}
    config['additive_gaussian_noise'] = {'stddev_range': [0, 10]}
    config['additive_speckle_noise'] = {'prob_range': [0, 0.0035]}
    config['additive_shade'] = {'transparency_range': [-0.5, 0.5], 'kernel_size_range': [100, 150]}
    config['motion_blur'] = {'max_kernel_size': 3}

    with tf.name_scope('online_augmentation'):
        prim_configs = [config.get(p, {}) for p in primitives]

        indices = tf.range(len(primitives))
        if random_order:
            indices = tf.random.shuffle(indices)

        def step(i, image):
            fn_pairs = [(tf.equal(indices[i], j), lambda p=p, c=c: getattr(photaug, p)(image, **c))
                        for j, (p, c) in enumerate(zip(primitives, prim_configs))]
            image = tf.case(fn_pairs)
            return i + 1, image

        _, aug_image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)),
                                     step, [0, image], parallel_iterations=1)

    return aug_image

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
                    image = cv2.imread(data[2 * view])
                    cam = load_cam(open(data[2 * view + 1]))
                    images.append(image)
                    cams.append(cam)
                depth_image = load_pfm(open(data[2 * self.view_num]))

                # dataset specified process
                if FLAGS.train_blendedmvs:
                    # downsize by 4 to fit depth map output
                    depth_image = scale_image(depth_image, scale=FLAGS.sample_scale)
                    cams = scale_mvs_camera(cams, scale=FLAGS.sample_scale)

                elif FLAGS.train_dtu:
                    # set depth range to [425, 937]
                    cams[0][1, 3, 0] = 425
                    cams[0][1, 3, 3] = 937

                elif FLAGS.train_eth3d:
                    # crop images
                    images, cams, depth_image = crop_mvs_input(
                        images, cams, depth_image, max_w=FLAGS.max_w, max_h=FLAGS.max_h)
                    # downsize by 4 to fit depth map output
                    depth_image = scale_image(depth_image, scale=FLAGS.sample_scale)
                    cams = scale_mvs_camera(cams, scale=FLAGS.sample_scale)
                
                else:
                    print ('Please specify a valid training dataset.')
                    exit(-1)

                # fix depth range and adapt depth sample number 
                cams[0][1, 3, 2] = FLAGS.max_d
                cams[0][1, 3, 1] = (cams[0][1, 3, 3] - cams[0][1, 3, 0]) / FLAGS.max_d

                # mask out-of-range depth pixels (in a relaxed range)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                depth_image = mask_depth_image(depth_image, depth_start, depth_end)

                # return mvs input
                self.counter += 1
                duration = time.time() - start_time
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                print('Forward pass: d_min = %f, d_max = %f.' % \
                    (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                yield (images, cams, depth_image) 

                # return backward mvs input for GRU
                if FLAGS.regularization == 'GRU':
                    self.counter += 1
                    start_time = time.time()
                    cams[0][1, 3, 0] = cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]
                    cams[0][1, 3, 1] = -cams[0][1, 3, 1]
                    duration = time.time() - start_time
                    print('Back pass: d_min = %f, d_max = %f.' % \
                        (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
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
    if FLAGS.regularization == 'GRU':
        training_sample_size = training_sample_size * 2
    print ('Training sample number: ', training_sample_size)

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
                    # get data
                    images, cams, depth_image = training_iterator.get_next()

                    # photometric augmentation and image normalization 
                    arg_images = []
                    for view in range(0, FLAGS.view_num):
                        image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                        if FLAGS.online_augmentation:
                            image = tf.map_fn(online_augmentation, image, back_prop=False) 
                        image = tf.image.per_image_standardization(image)                   
                        arg_images.append(image)
                    images = tf.stack(arg_images, axis=1)

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
                    if FLAGS.regularization == '3DCNNs':

                        # initial depth map
                        depth_map, prob_map = inference(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # refinement
                        if FLAGS.refinement:
                            ref_image = tf.squeeze(
                                tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                            refined_depth_map = depth_refine(depth_map, ref_image, 
                                    FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                        else:
                            refined_depth_map = depth_map

                        # regression loss
                        loss0, less_one_temp, less_three_temp = mvsnet_regression_loss(
                            depth_map, depth_image, depth_interval)
                        loss1, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                            refined_depth_map, depth_image, depth_interval)
                        loss = (loss0 + loss1) / 2

                    elif FLAGS.regularization == 'GRU':

                        # probability volume
                        prob_volume = inference_prob_recurrent(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)
                    
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
            summary_writer = tf.summary.FileWriter(FLAGS.log_folder, sess.graph)

            # load pre-trained model
            if FLAGS.use_pretrain:
                pretrained_model_path = os.path.join(FLAGS.model_folder, FLAGS.regularization, 'model.ckpt')
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(sess, '-'.join([pretrained_model_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                    ('-'.join([pretrained_model_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
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
                    if step % (FLAGS.display * 10) == 0:
                        summary_writer.add_summary(out_summary_op, total_step)
                   
                    # save the model checkpoint periodically
                    if (total_step % FLAGS.snapshot == 0 or step == (training_sample_size - 1)):
                        model_folder = os.path.join(FLAGS.model_folder, FLAGS.regularization)
                        if not os.path.exists(model_folder):
                            os.mkdir(model_folder)
                        ckpt_path = os.path.join(model_folder, 'model.ckpt')
                        print(Notify.INFO, 'Saving model to %s' % ckpt_path, Notify.ENDC)
                        saver.save(sess, ckpt_path, global_step=total_step)
                    step += FLAGS.batch_size * FLAGS.num_gpus
                    total_step += FLAGS.batch_size * FLAGS.num_gpus

def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # Prepare all training samples
    if FLAGS.train_blendedmvs:
        sample_list = gen_blendedmvs_path(FLAGS.blendedmvs_data_root, mode='training')
    if FLAGS.train_dtu:
        sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root)
    if FLAGS.train_eth3d:
        sample_list = gen_eth3d_path(FLAGS.eth3d_data_root, mode='training')
    # Shuffle
    random.shuffle(sample_list)
    # Training entrance.
    train(sample_list)


if __name__ == '__main__':
    print ('Training MVSNet with totally %d views inputs (including reference view)' % FLAGS.view_num)
    tf.app.run()

