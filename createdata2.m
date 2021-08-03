
close all;
clear;

%this file is to create new images and labels in certain size and change in
%correct format for resnet


% url = 'http://download.tensorflow.org/example_images/flower_photos.tgz';
% downloadFolder = "D:\SungRung\mnist_SEG(Noise)\images\dataFlower\input";
% filename = fullfile(downloadFolder,'flower_dataset.tgz');
% 
% dataFolder = fullfile(downloadFolder,'flower_photos');
% if ~exist(dataFolder,'dir')
%     fprintf("Downloading Flowers data set (218 MB)... ")
%     websave(filename,url);
%     untar(filename,downloadFolder)
%     fprintf("Done.\n")
% end

% %resizing labels edge detection ///////////////////////////////////////////
% OutputFolder = 'D:\SungRung\mnist_SEG(Noise)\images\0\noise';  % Set as needed [EDITED]
% dinfo = dir('D:\SungRung\mnist_SEG(Noise)\images\0\resized\*.jpg');% image extension
% % % D:\SungRung\mnist_SEG\images\0\orignal
% for K = 1 :length(dinfo)
%     thisimage = dinfo(K).name;
%     newName = "image9000" + K + '.png';
%     cd 'D:\SungRung\mnist_SEG(Noise)\images\0\resized'
%     Img   = imread(thisimage);
%     cd ..
%     i = imresize(Img, [720, 960], 'bilinear');
%     i = rgb2gray(i);
%     imshow()
%     imwrite(i, fullfile(OutputFolder, newName));  % [EDITED]
% end
% % 

OutputFolder = 'D:\SungRung\mnist_SEG(Noise)\images\0\nextTest';  % Set as needed [EDITED]
dinfo = dir('D:\SungRung\mnist_SEG(Noise)\images\0\resized\*.jpg')

for K = 1:length(dinfo)
    thisimage = dinfo(K).name;
    cd 'D:\SungRung\mnist_SEG(Noise)\images\0\resized';
    input   = imread(thisimage);
    cd ..
%     imshow(input);
%     pause;
%     input=double(input)+0.2;
%     Noise_Image=input * 0;
      i1 = double(input);
      epsilon=0.002;
     i1=i1+epsilon;
     i1 = imnoise(i1,'speckle',20);
     i2 = imnoise(i1, 'salt & pepper', 0.5);
     i3 = imnoise(i2, 'gaussian', 0.5);
%      i3=imadjust(i3,[0.65 0.7]);
%         Noise_Image=(Noise_Image+i3)/2;
%         
% 
%     end
%     subplot(1,4,4);
% 111
%     imshow(i3);
%      pause;
    imwrite(i3, fullfile(OutputFolder, thisimage))  % [EDITED]

end







%/////////////////////////////////////////////////////////////////////////////////////////////////






%resizing images and improving image data set ///////////////////////////
% OutputFolder = 'D:\SungRung\mnist_SEG(Noise)\images\0\speckle';  % Set as needed [EDITED]
% dinfo = dir('D:\SungRung\mnist_SEG(Noise)\images\0\resized\*.png');% image extension
% 
% for K = 1:length(dinfo)+1000
%     thisimage = dinfo(K).name;
%     cd 'D:\SungRung\mnist_SEG(Noise)\images\0\resized'
%     input   = imread(thisimage);
%     cd ..
%     
%     input=double(input)+0.2;
% 
% %     imshow(input)
%     Noise_Image=input * 0;
%     
% %     for k = 1: 10
%         i1 = imnoise(input,'speckle',3);
% %         subplot(1,4,2)
% %         imshow(i1)
%         i2 = imnoise(i1, 'salt & pepper', 0.5);
% %         subplot(1,4,3)
% %         imshow(i2)
%         i3 = imnoise(i2, 'gaussian', 0.5);
% %         Noise_Image=(Noise_Image+i3)/2;
% %         
% % 
% %     end
% %     subplot(1,4,4);
%     imshow(i3);
% %      pause;
%     imwrite(i3, fullfile(OutputFolder, thisimage))  % [EDITED]
% 
% end
% 
% % 
% %     input = repmat(i, [1 1 3]);
%     i = (input + 50);
%     for k = 1: 10
%         i1 = imnoise(i,'speckle',0.2);
%         i2 = imnoise(i1, 'salt & pepper', 0.2);
%         i3 = imnoise(i2, 'gaussian', 0.2);
%         newImage = i3;
%         input = (input + newImage) /1.5;
%         imshow(input)
%         pause;
%     end
% %     input = input/10;
%     imshow(input);
%     pause;




  
%     i = (i + 50);
% %     subplot(1,3,1);
%     imshow(i)
%     for k = 1: 20
%         i1 = imnoise(i,'speckle',0.2);
%         i2 = imnoise(i1, 'salt & pepper', 0.2);
%         i3 = imnoise(i2, 'gaussian', 0.2);
%         newImage = i3;
%         for num1  = 1:size(i,1)
%            for num2 = 1:size(i,2)
%                if newImage(num1, num2) < i(num1, num2)
%                   i(num1, num2) = newImage(num1,num2); 
%                end
% %                1
%            end
%            
%         end
%     end
%     pause;
%     imshow(i);
%     pause;

% %resizing labels edge detection ///////////////////////////////////////////
% OutputFolder = 'D:\SungRung\mnist_SEG\labels\0\resized';  % Set as needed [EDITED]
% dinfo = dir('D:\SungRung\mnist_SEG\images\0\resized\*.png');% image extension
% % % D:\SungRung\mnist_SEG\images\0\orignal
% for K = 1 :length(dinfo)
%     thisimage = dinfo(K).name;
%     cd 'D:\SungRung\mnist_SEG\labels\0\resized\'
%     Img   = imread(thisimage);
%     cd ..
% %     i = imresize(Img, [720, 960], 'bilinear');
%     i = edge(Img,'Canny');
% %     i= repmat(i, [1 1 3]);
%     imwrite(i, fullfile(OutputFolder, thisimage));  % [EDITED]
% end
% 
% 
% %     i =  edge(Img,'Canny');
% %resizing images
% function cmap = camvidColorMap()
%     cmap = [0 0 0
%             255 255 255
%            ];
% 
%     cmap = cmap ./ 255;
% end
