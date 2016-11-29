#!/usr/bin/env python
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

import numpy as np
import sys
import PIL
from PIL import Image
import os.path 
import random
from random import randint

try:
	os.mkdir('train_colors2')
except OSError:
	print('output folder already exists')
try:
	os.mkdir('train_depths2')
except OSError:
	print('output folder already exists')

counter = 1
for file in os.listdir("train_colors"):
    if file.endswith(".png"):
	depthFile = file.replace('rgb','depth')
	filePath = 'train_colors/' + file
	depthFilePath = 'train_depths/' + depthFile
	print(str(counter) + filePath + ' ' + depthFilePath)
	counter += 1	
	colorOriginal = Image.open(filePath)
	depthOriginal = Image.open(depthFilePath)

	rotation_std = 5.0


	filename = os.path.splitext(file)[0]
	depthFilename = os.path.splitext(depthFile)[0]

	for i in range(5):	

		color = colorOriginal
		depth = depthOriginal

		width, height = 561, 427
	 	newWidth, newHeight = 420, 320
		borderX = (width - newWidth) / 2
		borderY = (height - newHeight) / 2
		if randint(0,2) == 0:
			randomTranslationX = 0
			randomTranslationY = 0
			randomAngle = np.random.normal(0.0, rotation_std)
			color = color.rotate(randomAngle)
			depth = depth.rotate(randomAngle)
		else:
			randomScale = random.uniform(0.75, 1.25)
			resizeWidth, resizeHeight = int(randomScale * width), int(randomScale * height)
			color = color.resize((resizeWidth, resizeHeight), PIL.Image.ANTIALIAS)
			depth = depth.resize((resizeWidth, resizeHeight), PIL.Image.ANTIALIAS)
			depthArray = np.array(depth)
			depthArray = depthArray.astype(np.float32)
			depthArray /= randomScale
			depthArray = np.clip(depthArray, 0.0, 255.0)
			depthArray = depthArray.astype(np.uint8)
			depth = Image.fromarray(depthArray)

			width, height = color.size
			borderX = (width - newWidth) / 2
			borderY = (height - newHeight) / 2

			if borderX <= 1:
				randomTranslationX = 0
				randomTranslationY = 0
			else:
				randomTranslationX = randint(-borderX + 1,borderX-1)
				randomTranslationY = randint(-borderY + 1,borderY-1)
		
		colorNew = color.crop((borderX + randomTranslationX, borderY + randomTranslationY,width - borderX + randomTranslationX, height - borderY + randomTranslationY))
		depthNew = depth.crop((borderX + randomTranslationX, borderY + randomTranslationY,width - borderX + randomTranslationX, height - borderY + randomTranslationY))

		colorArray = np.array(colorNew)
		colorArray = colorArray.astype(np.float32) / 255.0
		colorArray = matplotlib.colors.rgb_to_hsv(colorArray)
		randomHueShift = random.uniform(-0.1,0.1)
		colorArray[:,:,0] += randomHueShift
		colorArray[:,:,0] = np.mod(colorArray[:,:,0], 1.0)
		randomSaturationShift = random.uniform(-0.1,0.1)
		colorArray[:,:,1] += randomSaturationShift
		colorArray[:,:,1] = np.clip(colorArray[:,:,1], 0, 1)
		randomValueShift = random.uniform(-0.1,0.1)
		colorArray[:,:,2] += randomValueShift
		colorArray[:,:,2] = np.clip(colorArray[:,:,2], 0, 1)
		colorArray = matplotlib.colors.hsv_to_rgb(colorArray) * 255.0

		randomContrastChange = random.uniform(175.0,335.0)
		colorArray *= randomContrastChange / 255.0
		colorArray -= (randomContrastChange - 255.0) / 2.0
		colorArray = np.clip(colorArray, 0, 255.0)
		colorArray = colorArray.astype(np.uint8)
		colorNew = Image.fromarray(colorArray)	

		colorNew.save('train_colors2/' + filename + str(i) + '.png')
		depthNew.save('train_depths2/' + depthFilename + str(i) + '.png')
		
		colorNewH = colorNew.transpose(PIL.Image.FLIP_LEFT_RIGHT)
		depthNewH = depthNew.transpose(PIL.Image.FLIP_LEFT_RIGHT)
		colorNewH.save('train_colors2/' + filename + str(i) + 'f.png')
		depthNewH.save('train_depths2/' + depthFilename + str(i) + 'f.png')


