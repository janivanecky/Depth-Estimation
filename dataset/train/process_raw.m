% Master's Thesis - Depth Estimation by Convolutional Neural Networks
% Jan Ivanecky; xivane00@stud.fit.vutbr.cz

addpath('tools');

d = dir('.');
isub = [d(:).isdir]; %# returns logical vector
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..','tools'})) = [];
nameFolds(~cellfun(@isempty,(regexp(nameFolds,'._out')))) = [];
disp(numel(nameFolds));

count = 0;
outCount = 0;
for f = 1:numel(nameFolds)
        disp(f);
	disp(nameFolds{f});
	files = get_synched_frames(nameFolds{f});
        c = numel(files);
	disp(strcat('filecount: ',int2str(c)));

	files = files(1:5:c);
	c = numel(files);
	disp(strcat('filecount to process: ',int2str(c)));
	count = count + c;

	outFolder = strcat(nameFolds{f}, '_out');
	if ~exist(outFolder, 'dir')
		mkdir(outFolder);
	end
	parfor idx = 1:c
	    rgbFilename = strcat(nameFolds{f},'/',files(idx).rawRgbFilename);
	    depthFilename = strcat(nameFolds{f},'/',files(idx).rawDepthFilename);
	    outRGBFilename = strcat(nameFolds{f},'_out/',nameFolds{f},num2str(idx),'rgb.png');
	    outDepthFilename = strcat(nameFolds{f},'_out/',nameFolds{f},num2str(idx),'depth.png');
	    disp(outRGBFilename);
	    rgb = imread(rgbFilename);
	    depth = imread(depthFilename);
	    depth = swapbytes(depth);%
	    [depthOut, rgbOut] = project_depth_map(depth, rgb);
   	    imgDepth = fill_depth_colorization(double(rgbOut) / 255.0, depthOut, 0.8);
	    imgDepth = imgDepth / 10.0;
	    imgDepth = crop_image(imgDepth);
	    rgbOut = crop_image(rgbOut);
	    imwrite(rgbOut, outRGBFilename);
	    imwrite(imgDepth, outDepthFilename);
		
	end
        D = dir([outFolder, '/*rgb.png']);
	Num = length(D);%D(not([D.isdir])));
	disp(strcat('output filecount: ',int2str(Num)));
	outCount = outCount + Num;
end
disp(count);
disp(outCount);
     		
exit;
