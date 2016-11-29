#! /bin/bash
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

 
#Data 0

for f in train_colors0/*.png
do
	file=$(basename $f)
	depthfile="${file/rgb/depth}"
	printf "$file $depthfile\n" >> list_ordered.txt
		
done

sort --random-sort -o list.txt list_ordered.txt

awk < list.txt '{printf $1; printf " 0\n"}' > list_color.txt
awk < list.txt '{printf $2; printf " 0\n"}' > list_depth.txt

rm list.txt
rm list_ordered.txt

mkdir train_raw0_lmdb
$1/build/tools/convert_imageset -resize_height 27 -resize_width 37 -gray train_depths0/ list_depth.txt train_raw0_lmdb/train_raw0_depth_37x27.lmdb
$1/build/tools/convert_imageset -resize_height 54 -resize_width 74 -gray train_depths0/ list_depth.txt train_raw0_lmdb/train_raw0_depth_74x54.lmdb
$1/build/tools/convert_imageset -resize_height 218 -resize_width 298 train_colors0/ list_color.txt train_raw0_lmdb/train_raw0_color_298x218.lmdb

rm list_color.txt
rm list_depth.txt


#Data 1

for f in train_colors1/*.png
do
	file=$(basename $f)
	depthfile="${file/rgb/depth}"
	printf "$file $depthfile\n" >> list_ordered.txt
		
done

sort --random-sort -o list.txt list_ordered.txt

awk < list.txt '{printf $1; printf " 0\n"}' > list_color.txt
awk < list.txt '{printf $2; printf " 0\n"}' > list_depth.txt

rm list.txt
rm list_ordered.txt

mkdir train_raw1_lmdb
$1/build/tools/convert_imageset -resize_height 27 -resize_width 37 -gray train_depths1/ list_depth.txt train_raw1_lmdb/train_raw1_depth_37x27.lmdb
$1/build/tools/convert_imageset -resize_height 54 -resize_width 74 -gray train_depths1/ list_depth.txt train_raw1_lmdb/train_raw1_depth_74x54.lmdb
$1/build/tools/convert_imageset -resize_height 218 -resize_width 298 train_colors1/ list_color.txt train_raw1_lmdb/train_raw1_color_298x218.lmdb

rm list_color.txt
rm list_depth.txt

#Data 1

for f in train_colors2/*.png
do
	file=$(basename $f)
	depthfile="${file/rgb/depth}"
	printf "$file $depthfile\n" >> list_ordered.txt
		
done

sort --random-sort -o list.txt list_ordered.txt

awk < list.txt '{printf $1; printf " 0\n"}' > list_color.txt
awk < list.txt '{printf $2; printf " 0\n"}' > list_depth.txt

rm list.txt
rm list_ordered.txt

mkdir train_raw2_lmdb
$1/build/tools/convert_imageset -resize_height 27 -resize_width 37 -gray train_depths2/ list_depth.txt train_raw2_lmdb/train_raw2_depth_37x27.lmdb
$1/build/tools/convert_imageset -resize_height 54 -resize_width 74 -gray train_depths2/ list_depth.txt train_raw2_lmdb/train_raw2_depth_74x54.lmdb
$1/build/tools/convert_imageset -resize_height 218 -resize_width 298 train_colors2/ list_color.txt train_raw2_lmdb/train_raw2_color_298x218.lmdb

rm list_color.txt
rm list_depth.txt





