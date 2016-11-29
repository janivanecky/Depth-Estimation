===================================================
Required libraries for python2.7:
===================================================
- caffe, h5py, scipy, scikit-image, numpy, pypng and joblib.


===================================================
How to process the training dataset:
===================================================
1.) Download RAW NYU Depth v2. dataset (450GB) from http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip 
2.) Extract the RAW dataset into a folder A (name not important)
3.) Download NYU Depth v2. toolbox from http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip
4.) Extract scripts from the toolbox to folder 'tools' in folder A
5.) Run process_raw.m in folder A
6.) Download labeled NYU Depth v2. dataset from http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
7.) Download splits.mat containing official train/test split http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
8.) Make sure that labeled dataset and splits.mat are in the same folder, let's call it folder B
9.) Run get_train_scenes.m in the folder B
10.) Run split_train_set.sh in the folder B and pass it a single argument, path to folder A ('......./path/to/folder/A')
11.) Run scripts train_augment0.py, train_augment1.py, train_augment2.py in folder B
11.) Run create_train_lmdb.sh in folder B and pass it a path to caffe folder as an argument
12.) You should now have folders 'train_raw0_lmdb' (dataset version Data0), 'train_raw1_lmdb' (dataset version Data1), 'train_raw2_lmdb' (dataset version Data2) in folder B
*Note: all referenced scripts can be foun in folder 'train'

===================================================
How to process the testing dataset:
===================================================

1.) Download labeled NYU Depth v2. dataset from http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
2.) Download splits.mat containing official train/test split http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
3.) Place all downloaded files into single folder
4.) Run script process_test.sh
5.) Run create_test_lmdb.sh and pass it a path to caffe folder as an argument
6.) You should now have a folder 'test_lmdb' in your working directory
*Note: all referenced scripts can be found in folder 'test'
*Note2: files crop.py, _structure_classes.py, _solarized.py come from https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset
