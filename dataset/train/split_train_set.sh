# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

echo "Extracting images from training scenes..."
mkdir -p train_data_all
while read i; do
  	dirpath=${1}'/'${i}
	for d in "${dirpath}"*/
	do
		cp $d/* -t train_data_all
	done
done < train_scenes.txt

echo "Moving extracted RGB images to train_rgbs..."
mkdir -p train_colors
find train_data_all -name '*rgb.png' -exec mv -t train_colors {} +
echo "Moving extracted Depth images to train_depths..."
mkdir -p train_depths
find train_data_all -name '*depth.png' -exec mv -t train_depths {} +
rm -r train_data_all

