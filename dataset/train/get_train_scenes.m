% Master's Thesis - Depth Estimation by Convolutional Neural Networks
% Jan Ivanecky; xivane00@stud.fit.vutbr.cz

data = load('nyu_depth_v2_labeled.mat');
split = load('splits.mat');

for i = 1 : 795
	folders(i) = data.scenes(split.trainNdxs(i));
end

folders = unique(folders);
fileID = fopen('train_scenes.txt','w');

for i = 1 :numel(folders)
	fprintf(fileID, '%s\n', folders{i});
end	

fclose(fileID);
exit();
