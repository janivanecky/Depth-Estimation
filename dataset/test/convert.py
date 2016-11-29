#!/usr/bin/env python
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
# vim: set fileencoding=utf-8 :
#
# Helper script to convert the NYU Depth v2 dataset Matlab file into a set of
# PNG images in the CURFIL dataset format.
#
# See https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset

from __future__ import print_function

from joblib import Parallel, delayed
from skimage import exposure
from skimage.io import imsave
import h5py
import numpy as np
import os
import png
import scipy.io
import sys

from _structure_classes import get_structure_classes
import _solarized


def process_ground_truth(ground_truth):
    colors = dict()
    colors["structure"] = _solarized.colors[5]
    colors["prop"] = _solarized.colors[8]
    colors["furniture"] = _solarized.colors[9]
    colors["floor"] = _solarized.colors[1]
    shape = list(ground_truth.shape) + [3]
    img = np.ndarray(shape=shape, dtype=np.uint8)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            l = ground_truth[i, j]
            if (l == 0):
                img[i, j] = (0, 0, 0)  # background
            else:
                name = classes[names[l - 1]]
                assert name in colors, name
                img[i, j] = colors[name]
    return img


def visualize_depth_image(data):

    data[data == 0.0] = np.nan

    maxdepth = np.nanmax(data)
    mindepth = np.nanmin(data)
    data = data.copy()
    data -= mindepth
    data /= (maxdepth - mindepth)

    gray = np.zeros(list(data.shape) + [3], dtype=data.dtype)
    data = (1.0 - data)
    gray[..., :3] = np.dstack((data, data, data))

    # use a greenish color to visualize missing depth
    gray[np.isnan(data), :] = (97, 160, 123)
    gray[np.isnan(data), :] /= 255

    gray = exposure.equalize_hist(gray)

    # set alpha channel
    gray = np.dstack((gray, np.ones(data.shape[:2])))
    gray[np.isnan(data), -1] = 0.5

    return gray * 255


def convert_image(i, scene, img_depth, image, label):

    idx = int(i) + 1
    if idx in train_images:
        train_test = "training"
    else:
        assert idx in test_images, "index %d neither found in training set nor in test set" % idx
        train_test = "testing"

    folder = "%s/%s/%s" % (out_folder, train_test, scene)
    if not os.path.exists(folder):
        os.makedirs(folder)

    img_depth *= 1000.0

    png.from_array(img_depth, 'L;16').save("%s/%05d_depth.png" % (folder, i))

    depth_visualization = visualize_depth_image(img_depth)

    # workaround for a bug in the png module
    depth_visualization = depth_visualization.copy()  # makes in contiguous
    shape = depth_visualization.shape
    depth_visualization.shape = (shape[0], np.prod(shape[1:]))

    depth_image = png.from_array(depth_visualization, "RGBA;8")
    depth_image.save("%s/%05d_depth_visualization.png" % (folder, i))

    imsave("%s/%05d_colors.png" % (folder, i), image)

    ground_truth = process_ground_truth(label)
    imsave("%s/%05d_ground_truth.png" % (folder, i), ground_truth)


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: %s <h5_file> <train_test_split> <out_folder> [<rawDepth> <num_threads>]" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    h5_file = h5py.File(sys.argv[1], "r")
    # h5py is not able to open that file. but scipy is
    train_test = scipy.io.loadmat(sys.argv[2])
    out_folder = sys.argv[3]
    if len(sys.argv) >= 5:
        raw_depth = bool(int(sys.argv[4]))
    else:
        raw_depth = False

    if len(sys.argv) >= 6:
        num_threads = int(sys.argv[5])
    else:
        num_threads = -1

    test_images = set([int(x) for x in train_test["testNdxs"]])
    train_images = set([int(x) for x in train_test["trainNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    if raw_depth:
        print("using raw depth images")
        depth = h5_file['rawDepths']
    else:
        print("using filled depth images")
        depth = h5_file['depths']

    print("reading", sys.argv[1])

    labels = h5_file['labels']
    images = h5_file['images']

    rawDepthFilenames = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['rawDepthFilenames'][0]]
    names = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['names'][0]]
    scenes = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]
    rawRgbFilenames = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['rawRgbFilenames'][0]]
    classes = get_structure_classes()

    print("processing images")
    if num_threads == 1:
        print("single-threaded mode")
        for i, image in enumerate(images):
            print("image", i + 1, "/", len(images))
            convert_image(i, scenes[i], depth[i, :, :].T, image.T, labels[i, :, :].T)
    else:
        Parallel(num_threads, 5)(delayed(convert_image)(i, scenes[i], depth[i, :, :].T, images[i, :, :].T, labels[i, :, :].T) for i in range(len(images)))

    print("finished")
