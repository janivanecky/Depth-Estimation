#!/usr/bin/env python
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

from __future__ import print_function	
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2
import cv
import os.path
os.environ['GLOG_minloglevel'] = '2' 
import caffe
import scipy.ndimage
import argparse
import operator	
import shutil

from eval_grad import Test, PrintTop5

filter = np.zeros((1,3,3))
filter[0,0,:] = (-1,-1,-1)
filter[0,1,:] = (0,0,0)
filter[0,2,:] = (1,1,1)

filter2 = np.zeros((1,3,3))
filter2[0,0,:] = (-1,0,1)
filter2[0,1,:] = (-1,0,1)
filter2[0,2,:] = (-1,0,1)

WIDTH = 298
HEIGHT = 218
OUT_WIDTH = 35
OUT_HEIGHT = 25
GT_WIDTH = 418
GT_HEIGHT = 318

def filterImage(net, gt):
	net.blobs['X'].data[...] = gt
	net.forward()
	return (net.blobs['out'].data[0,0,:,:], net.blobs['out'].data[0,1,:,:])

def testNet(net, img):	
	net.blobs['X'].data[...] = img	
	net.forward()
	output = net.blobs['gradient'].data
	output = np.reshape(output, (1,2,OUT_HEIGHT, OUT_WIDTH))
	out1 = output[0,0,:,:]
	out2 = output[0,1,:,:]
	return out1, out2
	
def loadImage(path, channels, width, height):
	img = caffe.io.load_image(path)
	img = caffe.io.resize(img, (height, width, channels))
	img = np.transpose(img, (2,0,1))
	img = np.reshape(img, (1,channels,height,width))
	return img

def printImage(img, name, channels, width, height):
	params = list()
	params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
	params.append(8)

	imgnp = np.reshape(img, (height,width, channels))
	imgnp = np.array(imgnp * 255, dtype = np.uint8)
	cv2.imwrite(name, imgnp, params)

def eval(out, gt, rawResults):
		linearGT = gt#np.exp((gt - 1) / 0.179581) * 65.535
		linearOut = out#np.exp((out - 1) / 0.179581) * 65.635
		#RAW PIXEL TESTS
		rawResults = [x + y for x, y in zip(rawResults, Test(linearOut, linearGT))]
		return rawResults

			
caffe.set_mode_cpu()

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="directory with input images")
parser.add_argument("gt_dir", help="directory with ground truths")
parser.add_argument("output", help="folder to output to")
parser.add_argument("snaps", help="folder with snapshots to use")
args = parser.parse_args()


gradNet = caffe.Net("filter.prototxt", caffe.TEST)
gradNet.params['gradientFilter'][0].data[0,...] = filter
gradNet.params['gradientFilter'][0].data[1,...] = filter2

try:
	os.mkdir(args.output)
except OSError:
	x = 12
fileCount = len([name for name in os.listdir(args.input_dir)])

results = [dict() for x in range(2)]
for snapshot in os.listdir(args.snaps):
	if not snapshot.endswith("caffemodel"):
		continue
	currentSnapDir = snapshot.replace(".caffemodel","")
	if os.path.exists(args.output + "/" + currentSnapDir):
		shutil.rmtree(args.output + "/" + currentSnapDir)
	os.mkdir(args.output + "/" + currentSnapDir)
	print(currentSnapDir)
	sys.stdout.flush()
	netFile = snapshot.replace(".caffemodel",".prototxt")
	net = caffe.Net(args.snaps + '/' + netFile, args.snaps + '/' + snapshot, caffe.TEST)
	
	
	rawResults = np.zeros((2))
	for count, file in enumerate(os.listdir(args.input_dir)):
		out_string = str(count) + '/' + str(fileCount) + ': ' + file
		sys.stdout.write('%s\r' % out_string)
		sys.stdout.flush()
	
		inputFileName = file
		inputFilePath = args.input_dir + '/' + inputFileName
		gtFileName = file.replace('colors', 'depth')	
		gtFilePath = args.gt_dir + '/' + gtFileName
	
		gt = loadImage(gtFilePath, 1, GT_WIDTH + 2, GT_HEIGHT + 2)

		gt1, gt2 = filterImage(gradNet, gt)
		gt1 = np.reshape(gt1, (1,1,GT_HEIGHT, GT_WIDTH))
		gt2 = np.reshape(gt2, (1,1,GT_HEIGHT, GT_WIDTH))

		input = loadImage(inputFilePath, 3, WIDTH, HEIGHT)
				
		input *= 255
		input -= 127

		out1, out2 = testNet(net, input)
	
		outWidth = OUT_WIDTH
		outHeight = OUT_HEIGHT
		scaleW = float(GT_WIDTH) / float(OUT_WIDTH)
		scaleH = float(GT_HEIGHT) / float(OUT_HEIGHT)
		out1 = scipy.ndimage.zoom(out1, (scaleH,scaleW), order=3)
		out2 = scipy.ndimage.zoom(out2, (scaleH,scaleW), order=3)
		outWidth *= scaleW
		outHeight *= scaleH

	
		rawResults = eval(out1, gt1, rawResults)
		rawResults = eval(out2, gt2, rawResults)

		gt1 = (gt1 - gt1.min())/(gt1.max() - gt1.min())
		gt2 = (gt2 - gt2.min())/(gt2.max() - gt2.min())
		out1 -= out1.mean()
		out1 /= out1.std()
		out1 *= gt1.std()
		out1 += gt1.mean()
		out2 -= out2.mean()
		out2 /= out2.std()
		out2 *= gt2.std()
		out2 += gt2.mean()		
		input += 127
		input = input / 255.0
		input = np.transpose(input, (0,2,3,1))
		input = input[:,:,:,(2,1,0)]
		gt1 = np.clip(gt1, 0, 1)		
		gt2 = np.clip(gt2, 0, 1)
		out1 = np.clip(out1, 0, 1)
		out2 = np.clip(out2, 0, 1)
	
		filename = os.path.splitext(os.path.basename(inputFileName))[0]
		filePath = args.output + '/' + currentSnapDir + '/' + filename + '.png'
		printImage(input, filePath, 3, WIDTH, HEIGHT)
		printImage(out1, filePath.replace('_colors','_grad1'), 1, outWidth, outHeight)
		printImage(out2, filePath.replace('_colors','_grad2'), 1, outWidth, outHeight)
		printImage(gt1, filePath.replace('_colors', '_gt1'), 1, outWidth, outHeight)
		printImage(gt2, filePath.replace('_colors', '_gt2'), 1, outWidth, outHeight)
			
	
	rawResults[:] = [x / (fileCount * 2.0) for x in rawResults]
	
	
	for i in xrange(2):
		results[i][currentSnapDir] = rawResults[i]
		
titles = ["RMSE", "MVN"]
for i in xrange(2):
		results[i] = sorted(results[i].items(), key=operator.itemgetter(1))
		PrintTop5(titles[i], results[i])
