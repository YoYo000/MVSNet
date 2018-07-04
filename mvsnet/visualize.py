import numpy as np
import cv2
import re
import argparse
import sys
import matplotlib.pyplot as plt

from preprocess import load_pfm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('disp_path')
    args = parser.parse_args()
    disp_path = args.disp_path
    if disp_path.endswith('npy'):
        disp = np.load(disp_path)
        disp = np.squeeze(disp)
        print('value range: ', disp.min(), disp.max())
        plt.imshow(disp, 'rainbow')
        plt.show()
    elif disp_path.endswith('pfm'):
        disp = load_pfm(open(disp_path))
        ma = np.ma.masked_equal(disp, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(disp, 'rainbow')
        plt.show()
    else:
        disp = cv2.imread(disp_path)
        ma = np.ma.masked_equal(disp, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(disp)
        plt.show()
