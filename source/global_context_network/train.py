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
solver.solve()

