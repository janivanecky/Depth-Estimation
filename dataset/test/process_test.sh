# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

python convert.py nyu_depth_v2_labeled.mat splits.mat out 0 1
mkdir -p test_data
find out/testing -name *colors.png  -exec mv -t test_data {} +
find out/testing -name *depth.png  -exec mv -t test_data {} +
mkdir -p test_colors
mkdir -p test_depths

python crop.py


