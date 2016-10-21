#!/usr/bin/env python
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2
import cv
import caffe
import operator
import argparse
import os

from scipy import misc
from os.path import basename


def RootMeanSquaredError(output, gt):
	d = output - gt
	diff = np.sqrt(np.mean(d * d))
	return diff

def MVNError(output, gt):

	outMean = np.mean(output)
	outStd = np.std(output)
	output = (output - outMean)/outStd		

	gtMean = np.mean(gt)
	gtStd = np.std(gt)
	gt = (gt - gtMean)/gtStd

	d = output - gt
	diff = np.sqrt(np.mean(d * d))
	return diff


def Test(out, gt):
	RMSE = RootMeanSquaredError(out, gt)
	MVN = MVNError(out, gt)
	return [RMSE, MVN]

def PrintTop5(title, result):
	length = min(10, len(result))
	print
	print
	print ("TOP " + str(length) + " for " + title)
	for i in xrange(length):
		print (str(i) + ". " + result[i][0] + ': ' +  str(result[i][1]))
	print
	print
