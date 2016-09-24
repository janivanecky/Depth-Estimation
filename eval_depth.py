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

def LogDepth(depth):
	depth = np.maximum(depth, 1.0 / 255.0)	
	return 0.179581 * np.log(depth) + 1
	
def AbsoluteRelativeDifference(output, gt):
	gt = np.maximum(gt, 1.0 / 255.0)
	diff = np.mean(np.absolute(output - gt) / gt)
	return diff	

def SquaredRelativeDifference(output, gt):
	gt = np.maximum(gt, 1.0 / 255.0)
	d = output - gt
	diff = np.mean((d * d) / gt)
	return diff	

def RootMeanSquaredError(output, gt):
	d = output - gt
	diff = np.sqrt(np.mean(d * d))
	return diff

def RootMeanSquaredErrorLog(output, gt):
	d = LogDepth(output / 10.0) * 10.0 - LogDepth(gt / 10.0) * 10.0
	diff = np.sqrt(np.mean(d * d))
	return diff

def ScaleInvariantMeanSquaredError(output, gt):
	output = LogDepth(output / 10.0) * 10.0
	gt = LogDepth(gt / 10.0) * 10.0
	d = output - gt
	diff = np.mean(d * d)

	relDiff = (d.sum() * d.sum()) / float(d.size * d.size)
	return diff - relDiff

def Log10Error(output, gt):
	output = np.maximum(output, 1.0 / 255.0)
	gt = np.maximum(gt, 1.0 / 255.0)
	diff = np.mean(np.absolute(np.log10(output) - np.log10(gt)))
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

def Threshold(output, gt, threshold):
	output = np.maximum(output, 1.0 / 255.0)
	gt = np.maximum(gt, 1.0 / 255.0)
	withinThresholdCount = np.where(np.maximum(output / gt, gt / output) < threshold)[0].size
	return withinThresholdCount / float(gt.size)

def Test(out, gt):
	absRelDiff = AbsoluteRelativeDifference(out, gt)	
	sqrRelDiff = SquaredRelativeDifference(out, gt)
	RMSE = RootMeanSquaredError(out, gt)
	RMSELog = RootMeanSquaredErrorLog(out, gt)
	SIMSE = ScaleInvariantMeanSquaredError(out, gt)
	threshold1 = Threshold(out, gt, 1.25)
	threshold2 = Threshold(out, gt, 1.25 * 1.25)
	threshold3 = Threshold(out, gt, 1.25 * 1.25 * 1.25)
	log10 = Log10Error(out, gt)
	MVN = MVNError(out, gt)
	return [absRelDiff, sqrRelDiff, RMSE, RMSELog, SIMSE, log10, MVN, threshold1, threshold2, threshold3]

def PrintTop5(title, result):
	length = min(10, len(result))
	print
	print
	print ("TOP " + str(length) + " for " + title)
	for i in xrange(length):
		print (str(i) + ". " + result[i][0] + ': ' +  str(result[i][1]))
	print
	print

