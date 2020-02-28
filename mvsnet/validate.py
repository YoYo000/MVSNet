#!/usr/bin/env python
"""
Copyright 2020, Yao Yao, HKUST.
Validatation script.
"""

from __future__ import print_function

import os
import time
import sys
import math
import argparse
import numpy as np

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append("../")
from tools.common import Notify
from preprocess import *
from model import *
from loss import *

# params for datasets
tf.app.flags.DEFINE_string('blendedmvs_data_root', '/data/BlendedMVS/dataset_low_res', 
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('eth3d_data_root', '/data/eth3d/lowres/training/undistorted', 
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('dtu_data_root', '/data/dtu', 
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('validate_set', 'dtu', 
                            """Dataset to validate.""")

# params for config
tf.app.flags.DEFINE_integer('view_num', 3, 
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 256, 
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 640, 
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 512, 
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25, 
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('interval_scale', 1, 
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """training batch size""")
tf.app.flags.DEFINE_bool('inverse_depth', False,
                           """Apply inverse depth.""")
tf.app.flags.DEFINE_string('regularization', '3DCNNs', 
                            """Regularization type.""")

# params for paths
tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', 
                           '/data/tf_model/3DCNNs/BlendedMVS/blended_augmented/model.ckpt',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 150000,
                            """ckpt step.""")
tf.app.flags.DEFINE_string('validation_result_path', 
                           '/data/tf_model/3DCNNs/BlendedMVS/blended_augmented/validation_results.txt',
                           """Path to restore the model.""")

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
                    image = cv2.imread(data[2 * view])
                    cam = load_cam(open(data[2 * view + 1]), FLAGS.interval_scale)
                    cam[1, 3, 1] = (cam[1, 3, 3] - cam[1, 3, 0]) / FLAGS.max_d
                    cam[1, 3, 2] = FLAGS.max_d
                    images.append(image)
                    cams.append(cam)
                depth_image = load_pfm(open(data[2 * self.view_num]))

                if FLAGS.validate_set == 'eth3d':
                    # crop to fit the network
                    images, cams, depth_image = crop_mvs_input(
                        images, cams, depth_image, max_w=FLAGS.max_w, max_h=FLAGS.max_h)
                    # downsize by 4 to fit depth map output
                    cams = scale_mvs_camera(cams, scale=FLAGS.sample_scale)
                    depth_image = scale_image(depth_image, scale=FLAGS.sample_scale)

                if FLAGS.validate_set == 'blendedmvs':
                    # downsize by 4 to fit depth map output
                    depth_image = scale_image(depth_image, scale=FLAGS.sample_scale)
                    cams = scale_mvs_camera(cams, scale=FLAGS.sample_scale)
                    
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                depth_image = mask_depth_image(depth_image, depth_start, depth_end)

                # return 
                self.counter += 1
                duration = time.time() - start_time
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                yield (images, cams, depth_image) 

def validate_mvsnet(mvs_list):
    """ validate mvsnet """
    print ('sample number: ', len(mvs_list))

    # Training and validation generators
    mvs_generator = iter(MVSGenerator(mvs_list, FLAGS.view_num))
    generator_data_type = (tf.float32, tf.float32, tf.float32)
    # Datasets from generators
    mvs_set = tf.data.Dataset.from_generator(lambda: mvs_generator, generator_data_type)
    mvs_set = mvs_set.batch(FLAGS.batch_size)
    mvs_set = mvs_set.prefetch(buffer_size=1)
    # iterators
    mvs_iterator = mvs_set.make_initializable_iterator()
    # data
    images, cams, depth_image = mvs_iterator.get_next()

    # consolidate inputs
    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    depth_image.set_shape(tf.TensorShape([None, None, None, 1]))
    depth_start = tf.reshape(tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), 
                             [FLAGS.batch_size])
    depth_interval = tf.reshape(tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), 
                                [FLAGS.batch_size])
    depth_num = tf.cast(tf.reshape(tf.slice(cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), 'int32')
    if FLAGS.inverse_depth:
        depth_end = tf.reshape(
            tf.slice(cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    else:
        depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # image normalization
    normalized_images = []
    for view in range(0, FLAGS.view_num):
        image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
        image = tf.image.per_image_standardization(image)
        normalized_images.append(image)
    images = tf.stack(normalized_images, axis=1)

    # depth map inference
    if FLAGS.regularization == '3DCNNs':
        depth_map, prob_map = inference(
            images, cams, FLAGS.max_d, depth_start, depth_interval)
    elif FLAGS.regularization == 'GRU':
        depth_map, prob_map = inference_winner_take_all(images, cams, 
            depth_num, depth_start, depth_end, reg_type='GRU', inverse_depth=FLAGS.inverse_depth)

    if FLAGS.inverse_depth:
        interval = tf.ones_like(depth_interval)
        loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
            depth_map, depth_image, interval)
    else:
        loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
            depth_map, depth_image, depth_interval)

    # init option
    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ave_loss = 0
    ave_per1 = 0
    ave_per3 = 0
    model_folder = os.path.split(FLAGS.pretrained_model_ckpt_path)[0]
    with tf.Session(config=config) as sess:     
        # initialization
        sess.run(init_op)

        total_step = 0
        # load model
        if FLAGS.pretrained_model_ckpt_path is not None:
            restorer = tf.train.Saver(tf.global_variables())
            restorer.restore(
                sess, '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
            print(Notify.INFO, 'Pre-trained model restored from %s' %
                  ('-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
            total_step = FLAGS.ckpt_step
    
        # training of one epoch
        sess.run(mvs_iterator.initializer)
        for step in range(len(mvs_list)):

            # run one batch
            start_time = time.time()
            try:
                out_loss, out_less_one, out_less_three, out_depth_map = sess.run([
                    loss, less_one_accuracy, less_three_accuracy, depth_map])
            except tf.errors.OutOfRangeError:
                print("all dense finished")  # ==> "End of dataset"
                break
            duration = time.time() - start_time
            print(Notify.INFO, 'depth map validation for %d, loss=%.3f, < 1 = %.3f, < 3 = %.3f. (%.3f sec/step)' 
                  % (step, out_loss, out_less_one, out_less_three, duration), 
                  Notify.ENDC)

            # save output
            ave_loss += out_loss
            ave_per1 += out_less_one
            ave_per3 += out_less_three
            total_step += 1

        ave_loss /= len(mvs_list)
        ave_per1 /= len(mvs_list)
        ave_per3 /= len(mvs_list)
        print ('ave_loss', ave_loss)
        print ('ave_per1', ave_per1)
        print ('ave_per3', ave_per3)
        with open(FLAGS.validation_result_path, 'a') as log_file:
            log_file.write('Model check point %d, L1 loss = %f, < 1 = %f, < 3 = %f \n' 
                           % (int(FLAGS.ckpt_step), float(ave_loss), float(ave_per1), float(ave_per3)))

def main(argv=None):
    """ program entrance """
    # gen validation list
    if FLAGS.validate_set == 'blendedmvs':
        sample_list = gen_blended_mvs_path(FLAGS.blendedmvs_data_root, mode='validation')
    elif FLAGS.validate_set == 'eth3d':
        sample_list = gen_eth3d_path(FLAGS.eth3d_data_root, mode='validation')
    elif FLAGS.validate_set == 'dtu':
        sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root, mode='validation')

    # inference
    validate_mvsnet(sample_list)

if __name__ == '__main__':
    tf.app.run()
