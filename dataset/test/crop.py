#!/usr/bin/env python
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import PIL
from PIL import Image
import cv2
import cv
import caffe
import argparse
import os.path 
import random
from random import randint

parser = argparse.ArgumentParser()
#parser.add_argument("color_folder", help="input folder")

args = parser.parse_args()

for file in os.listdir("test_data"):
    if file.endswith(".png"):
	filePath = 'test_data/' + file

	width, height = 640, 480
 	newWidth, newHeight = 420, 320

	borderX = (width - newWidth) / 2
	borderY = (height - newHeight) / 2

	img = Image.open(filePath)
	img = img.crop((borderX, borderY, width - borderX, height - borderY))


	print(filePath)

	if 'depth' in file:
		depthArray = np.array(img)
		depthArray = depthArray.astype(np.float32)
		depthArray /= 65535.0
		depthArray = np.clip(depthArray, 0.0039, 1)
		depthArray *= 6.5535 # 1 - 10 meters

		depthArray *= 255
		depthArray = depthArray.astype(np.uint8)
		depthNew = Image.fromarray(depthArray)		
		depthNew.save('test_depths/' + file)

	if 'colors' in file:
		img.save('test_colors/' + file)


		
		
		


	
