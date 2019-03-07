#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Depth map visualization.
"""

import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from preprocess import load_pfm
from depthfusion import read_gipuma_dmb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('depth_path')
    args = parser.parse_args()
    depth_path = args.depth_path
    if depth_path.endswith('npy'):
        depth_image = np.load(depth_path)
        depth_image = np.squeeze(depth_image)
        print('value range: ', depth_image.min(), depth_image.max())
        plt.imshow(depth_image, 'rainbow')
        plt.show()
    elif depth_path.endswith('pfm'):
        depth_image = load_pfm(open(depth_path))
        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(depth_image, 'rainbow')
        plt.show()
    elif depth_path.endswith('dmb'):
        depth_image = read_gipuma_dmb(depth_path)
        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(depth_image, 'rainbow')
        plt.show()
    else:
        depth_image = cv2.imread(depth_path)
        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(depth_image)
        plt.show()
