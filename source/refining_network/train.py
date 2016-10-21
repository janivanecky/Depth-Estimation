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
solver.net.copy_from(path_to_global_context_network_caffemodel)

gradPart = caffe.Net(path_to_gradient_network_definition_file, path_to_gradient_network_caffemodel, caffe.TEST)

params = gradPart.params.keys()
source_params = {pr: (gradPart.params[pr][0].data, gradPart.params[pr][1].data) for pr in params}
target_params = {pr: (solver.net.params[pr][0].data, solver.net.params[pr][1].data) for pr in params}

for pr in params:
    if pr == 'conv1':
	solver.net.params['conv1-grad'][1].data[...] = source_params [pr][1]  #biases
	solver.net.params['conv1-grad'][0].data[...] = source_params [pr][0]  #weights
    else:
	target_params[pr][1][...] = source_params [pr][1]  #bias
	target_params[pr][0][...] = source_params [pr][0]  #weights

alexNet = caffe.Net(path_to_gradient_network_definition_file, 'bvlc_alexnet.caffemodel', caffe.TEST)
solver.net.params['conv1-refine'][1].data[...] = alexNet.params['conv1'][1].data  #biases
solver.net.params['conv1-refine'][0].data[...] = alexNet.params['conv1'][0].data  #weights

solver.solve()

