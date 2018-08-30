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
import numpy as np

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append("../")
from tools.common import Notify
from preprocess import *
from model import *

FLAGS = tf.app.flags.FLAGS


# params for datasets
tf.app.flags.DEFINE_string('dense_folder', None, 
                           """Root path to dense folder.""")
# params for input
tf.app.flags.DEFINE_integer('view_num', 5,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('default_depth_start', 1,
                            """Start depth when training.""")
tf.app.flags.DEFINE_integer('default_depth_interval', 1, 
                            """Depth interval when training.""")
tf.app.flags.DEFINE_integer('max_d', 192, 
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 1152, 
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 864, 
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25, 
                            """Downsample scale for building cost volume (W and H).""")
tf.app.flags.DEFINE_float('interval_scale', 0.8, 
                            """Downsample scale for building cost volume (D).""")
tf.app.flags.DEFINE_integer('base_image_size', 32, 
                            """Base image size to fit the network.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """training batch size""")

# params for config
tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', 
                           '/data/dtu/tf_model/mvsnet_arxiv/model.ckpt',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 70000,
                            """ckpt step.""")

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
                
                # read input data
                images = []
                cams = []
                image_index = int(os.path.splitext(os.path.basename(data[0]))[0])
                selected_view_num = int(len(data) / 2)

                for view in range(min(self.view_num, selected_view_num)):
                    # image = cv2.imread(data[2 * view])
                    image_file = file_io.FileIO(data[2 * view], mode='r')
                    image = scipy.misc.imread(image_file, mode='RGB')
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # cam = load_cam(open(data[2 * view + 1]))
                    cam_file = file_io.FileIO(data[2 * view + 1], mode='r')
                    cam = load_cam(cam_file)
                    cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale
                    images.append(image)
                    cams.append(cam)

                if selected_view_num < self.view_num:
                    for view in range(selected_view_num, self.view_num):
                        # image = cv2.imread(data[0])
                        image_file = file_io.FileIO(data[0], mode='r')
                        image = scipy.misc.imread(image_file, mode='RGB')
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        # cam = load_cam(open(data[1]))
                        cam_file = file_io.FileIO(data[1], mode='r')
                        cam = load_cam(cam_file)
                        cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale
                        images.append(image)
                        cams.append(cam)

                # determine a proper scale to resize input 
                h_scale = float(FLAGS.max_h) / images[0].shape[0]
                w_scale = float(FLAGS.max_w) / images[0].shape[1]
                if h_scale > 1 or w_scale > 1:
                    print ("max_h, max_w should < W and H!")
                    exit()
                resize_scale = h_scale
                if w_scale > h_scale:
                    resize_scale = w_scale
                scaled_input_images, scaled_input_cams = scale_mvs_input(images, cams, scale=resize_scale)

                # crop to fit network
                croped_images, croped_cams = crop_mvs_input(scaled_input_images, scaled_input_cams)
                image_shape = croped_images[0].shape

                # center images
                centered_images = []
                for view in range(self.view_num):
                    centered_images.append(center_image(croped_images[view]))

                # sample cameras for building cost volume
                real_cams = np.copy(croped_cams) 
                scaled_cams = scale_mvs_camera(croped_cams, scale=FLAGS.sample_scale)

                # return mvs input
                scaled_images = []
                for view in range(self.view_num):
                    scaled_images.append(scale_image(croped_images[view], scale=FLAGS.sample_scale))
                scaled_images = np.stack(scaled_images, axis=0)
                croped_images = np.stack(croped_images, axis=0)
                scaled_cams = np.stack(scaled_cams, axis=0)
                self.counter += 1
                yield (scaled_images, centered_images, scaled_cams, real_cams, image_index) 

def mvsnet_pipeline(mvs_list):
    """ mvsnet in altizure pipeline """

    # create output folder
    print ('sample number: ', len(mvs_list))
    output_folder = os.path.join(FLAGS.dense_folder, 'depths_mvsnet')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # Training and validation generators
    mvs_generator = iter(MVSGenerator(mvs_list, FLAGS.view_num))
    generator_data_type = (tf.float32, tf.float32, tf.float32, tf.float32, tf.int32)    
    # Datasets from generators
    mvs_set = tf.data.Dataset.from_generator(lambda: mvs_generator, generator_data_type)
    mvs_set = mvs_set.batch(FLAGS.batch_size)
    # iterators
    mvs_iterator = mvs_set.make_initializable_iterator()
    # data
    croped_images, centered_images, scaled_cams, croped_cams, image_index = mvs_iterator.get_next()
    croped_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    centered_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    scaled_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    depth_start = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_interval = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])

    # depth map inference
    init_depth_map, prob_map = inference_mem(
        centered_images, scaled_cams, FLAGS.max_d, depth_start, depth_interval)

    # refinement 
    ref_image = tf.squeeze(tf.slice(centered_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    depth_map = depth_refine(init_depth_map, ref_image, FLAGS.max_d, depth_start, depth_interval)
                                            
    # init option
    init_op = tf.global_variables_initializer()
    var_init_op = tf.local_variables_initializer()
    # GPU grows incrementally
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:   

        # initialization
        sess.run(var_init_op)
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
    
        # run inference for each reference view
        sess.run(mvs_iterator.initializer)
        for step in range(len(mvs_list)):

            start_time = time.time()
            try:
                out_depth_map, out_init_depth_map, out_prob_map, out_images, out_cams, out_index = sess.run(
                    [depth_map, init_depth_map, prob_map, croped_images, scaled_cams, image_index])
            except tf.errors.OutOfRangeError:
                print("all dense finished")  # ==> "End of dataset"
                break
            duration = time.time() - start_time
            print(Notify.INFO, 'depth inference %d finished. (%.3f sec/step)' % (step, duration), 
                  Notify.ENDC)

            # squeeze output
            out_estimated_depth_image = np.squeeze(out_depth_map)
            out_init_depth_image = np.squeeze(out_init_depth_map)
            out_prob_map = np.squeeze(out_prob_map)
            out_ref_image = np.squeeze(out_images)
            out_ref_image = np.squeeze(out_ref_image[0, :, :, :])
            out_ref_cam = np.squeeze(out_cams)
            out_ref_cam = np.squeeze(out_ref_cam[0, :, :, :])
            out_index = np.squeeze(out_index)

            # paths
            depth_map_path = output_folder + ('/%08d.pfm' % out_index)
            init_depth_map_path = output_folder + ('/%08d_init.pfm' % out_index)
            prob_map_path = output_folder + ('/%08d_prob.pfm' % out_index)
            out_ref_image_path = output_folder + ('/%08d.jpg' % out_index)
            out_ref_cam_path = output_folder + ('/%08d.txt' % out_index)

            # save output
            write_pfm(init_depth_map_path, out_init_depth_image)
            write_pfm(depth_map_path, out_estimated_depth_image)
            write_pfm(prob_map_path, out_prob_map)
            out_ref_image = cv2.cvtColor(out_ref_image, cv2.COLOR_RGB2BGR)
            image_file = file_io.FileIO(out_ref_image_path, mode='w')
            scipy.misc.imsave(image_file, out_ref_image)
            write_cam(out_ref_cam_path, out_ref_cam)
            total_step += 1


def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # generate input path list
    mvs_list = gen_pipeline_mvs_list(FLAGS.dense_folder)
    # mvsnet inference
    mvsnet_pipeline(mvs_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dense_folder', type=str, default = FLAGS.dense_folder)
    parser.add_argument('--view_num', type=int, default = FLAGS.view_num)
    args = parser.parse_args()

    FLAGS.dense_folder = args.dense_folder
    FLAGS.view_num = args.view_num
    print ('Testing MVSNet with %d views' % args.view_num)

    tf.app.run()