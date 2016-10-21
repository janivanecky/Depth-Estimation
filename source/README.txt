================================================
Structure of this folder
================================================
	global_context_network - contains network definitions and example scripts for training and evaluating the the global context network from the proposed model

	gradient_network - contains network definitions and example scripts for training and evaluating the gradient network from the proposed model

	joint - contains network definitions and example scripts for training and evaluating the jointly trained global context network and gradient networks from the proposed model

	refining_network - contains network definitions and example scripts for training and evaluating the refining network from the proposed model

Each of these folders contains:

	-multiple subdirectories, each with different configuration/loss function of the network. Individual subdirectories contain the network definition files for the training - 'net_train.prototxt' and for evaluating - 'net_deploy.prototxt'.

	-script 'train.py' for training. Note that this script is just an example and it's content should be modified to fit the desired training process.

	-script 'eval_depth.py' or 'eval_grad.py', contains definitions of error functions used for evaluating the performance

	-script 'test_depth.py' or 'test_grad.py'. This script is used to evaluate the performance of the network and visualize it's output.

	-'solver.prototxt' - example of the definition file for the Caffe solver. 

================================================
Usage of the 'test_depth.py'/'test_grad.py' scripts:
================================================

	python test_depth.py INPUT_DIR GT_DIR OUT_DIR SNAPSHOTS_DIR [--log]

-INPUT_DIR is the path to the folder containing input images
-GT_DIR is the path to the folder containing ground truth depth maps
-OUT_DIR is the path to the folder to which will be written output depth maps
-SNAPSHOTS_DIR is the path to the folder containing .caffemodel files containing trained network models. All models from this folder will be evaluated.
--log switch is used when the depth values that are produced by the network are in log space

=================================================
Frameworks/Libraries needed:
================================================

Caffe
Python2.7:	
- caffe, scipy, scikit-image, numpy, pypng, cv2, Pillow, matplotlib

=================================================
Few notes
=================================================
	-input images should be named in a same way as the corresponding ground truths, with difference that input images should have a suffix 'colors', while ground truth images should have a suffix 'depth'. Note that these suffixes should preceed file extension, e.g., 'image1_colors.png' and corresponding depth map 'image1_depth.png'

	-along with .caffemodel file, corresponding deploy network definition file has to be placed into SNAPSHOTS_DIR, with the same name as the model file but with different extension 'prototxt' instead of 'caffemodel'

	-there will actually be two output folders created, one OUT_DIR and the other OUT_DIR + '_abs'. OUT_DIR contains output depths that are fit using MVN normalization onto ground truth, OUT_DIR + '_abs' contains the raw output depth maps.
	
	-note that you need AlexNet caffemodel for the training of the global context network, gradient network and their joint configuration. It can be downloaded here: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
