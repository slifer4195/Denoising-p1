clear all;
close all;

dataFolder2 = "D:\SungRung\mnist_SEG(Noise)\images\0\resultDog\combined";
imds2 = imageDatastore(dataFolder2,'IncludeSubfolders',true,'LabelSource','foldernames');
image = readimage(imds2,1);
% imshow(image);
% pause;
newImage=image * 0.0;
for i = 1: 3
    image = readimage(imds2, i);
%     imshow(image);
%     pause;
    newImage = (newImage + image);
end
% newImage = im2bw(newImage, 0.5);
imshow(newImage);
imwrite(newImage, fullfile(dataFolder2,"result2.png"));

% 
% OutputFolder = 'D:\SungRung\mnist_SEG(Noise)\images\0\comboNoise';  % Set as needed [EDITED]
% dinfo = dir('D:\SungRung\mnist_SEG(Noise)\images\0\resultsLisence\*.png')