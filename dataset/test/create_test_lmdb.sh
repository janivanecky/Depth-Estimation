#! /bin/bash
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

for f in test_colors/*.png
do
	file=$(basename $f)
	depthfile="${file/colors/depth}"
	printf "$file $depthfile\n" >> list_ordered.txt
		
done

sort --random-sort -o list.txt list_ordered.txt

awk < list.txt '{printf $1; printf " 0\n"}' > list_color.txt
awk < list.txt '{printf $2; printf " 0\n"}' > list_depth.txt

rm list.txt
rm list_ordered.txt

mkdir -p test_lmdb
$1/build/tools/convert_imageset -resize_height 27 -resize_width 37 -gray test_depths/ list_depth.txt test_lmdb/test_depth_37x27.lmdb
$1/build/tools/convert_imageset -resize_height 54 -resize_width 74 -gray test_depths/ list_depth.txt test_lmdb/test_depth_74x54.lmdb
$1/build/tools/convert_imageset -resize_height 218 -resize_width 298 test_colors/ list_color.txt test_lmdb/test_color_298x218.lmdb

rm list_depth.txt
rm list_color.txt


