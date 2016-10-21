#!/usr/bin/env python
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

import numpy as np
import cv2
import cv
import caffe
from caffe.proto import caffe_pb2
import sys

from google.protobuf import text_format
import argparse

caffe.set_mode_gpu()
solver = caffe.get_solver('solver.prototxt')
solver.net.copy_from('bvlc_alexnet.caffemodel')

solver.net.params['conv1_g'][0].data[...] = solver.net.params['conv1'][0].data
solver.net.params['conv1_g'][1].data[...] = solver.net.params['conv1'][1].data

filter = np.zeros((1,3,3))
filter[0,0,:] = (-1,-1,-1)
filter[0,1,:] = (0,0,0)
filter[0,2,:] = (1,1,1)

filter2 = np.zeros((1,3,3))
filter2[0,0,:] = (-1,0,1)
filter2[0,1,:] = (-1,0,1)
filter2[0,2,:] = (-1,0,1)

solver.net.params['gradientFilter'][0].data[0,...] = filter
solver.net.params['gradientFilter'][0].data[1,...] = filter2

solver.solve()

