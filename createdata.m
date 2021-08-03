
close all;
clear;

%this file is to create new images and labels in certain size and change in
% %correct format for resnet


%resizing images and improving image data set ///////////////////////////
OutputFolder = 'D:\SungRung\mnist_SEG(Noise)\images\0\comboNoise';  % Set as needed [EDITED]
dinfo = dir('D:\SungRung\mnist_SEG(Noise)\images\0\final\*.jpg');% image extension
cmap = camvidColorMap;
for K = 1:length(dinfo)
    thisimage = dinfo(K).name;
    cd 'D:\SungRung\mnist_SEG(Noise)\images\0\resized\'
    input   = imread(thisimage);
    cd ..
    i = (input + 50);
    o1 = imnoise(i,'speckle',0.2);
    o2 = imnoise(o1, 'salt & pepper', 0.4);
    o3 = imnoise(o2, 'gaussian', 0.2);
    subplot(1,2,1);
    imshow(o3);
    for k = 1: 1000
        i1 = imnoise(i,'speckle',0.2);
        i2 = imnoise(i1, 'salt & pepper', 0.2);
        i3 = imnoise(i2, 'gaussian', 0.2);
        input = (03 + i3) /2;

    end
    subplot(1,2,2);
    imshow(input);
   pause;
%     imwrite(i, fullfile(OutputFolder, thisimage));  % [EDITED]

end

% 
%     input = repmat(i, [1 1 3]);
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
