clear all;
close all;

% loading pretrainedNetwork
pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/deeplabv3plusResnet18CamVid.mat';
pretrainedFolder = fullfile(tempdir,'pretrainedNetwork');
pretrainedNetwork = fullfile(pretrainedFolder,'deeplabv3plusResnet18CamVid.mat'); 
if ~exist(pretrainedNetwork,'file')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained network (58 MB)...');
    websave(pretrainedNetwork,pretrainedURL);
end

%loading data
dataFolder = 'D:\SungRung\mnist_SEG(Noise)\images\0\nextTest';
imds = imageDatastore(dataFolder,'IncludeSubfolders',true,'LabelSource','foldernames',"ReadFcn",@increase);

[imdsTrain, imdsValid] = splitEachLabel(imds, 0.7, 'randomized');
numTrainImages = numel(imdsTrain.Labels);

 
cmap = camvidColorMap;
% %loading pixel-label images

classes = ["background" "edge"];
labelIDs = camvidPixelLabelIDs();
labelDir = fullfile('D:', "SungRung", "mnist_SEG(Noise)","images", "0", "final");
pxds =pixelLabelDatastore(labelDir, classes, labelIDs, 'ReadFcn', @to2D);

tbl = countEachLabel(pxds);

% for i = 1:100:10000
%     I = readimage(imds, i);
%     I = cast(I, 'double');
%     C = readimage(pxds,i);
%     CB= labeloverlay(I, C); 
%     subplot(1,2,1)
%     imshow(I);
%     subplot(1,2,2)
%     imshow(CB);
%     pause;
% end


[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds, pxds);

% [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds, pxds);
numTrainingImages = numel(imdsTrain.Files);
%numValImages = numel(imdsVal.Files);
numTestingImages = numel(imdsTest.Files);

% % % % % % % %create network

imageSize = [720 960 3];
numClasses = numel(classes);
lgraph = deeplabv3plusLayers(imageSize, numClasses, 'resnet18');
 
 %balance class weights
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./imageFreq;

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);


lgraph = removeLayers(lgraph, 'classification');
lgraph = addLayers(lgraph, pxLayer);

lgraph = connectLayers(lgraph, 'softmax-out', 'labels')


% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'MaxEpochs',1, ...  
    'MiniBatchSize',1, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress');
% data augmentation
% pause;
augmenter = imageDataAugmenter('RandXReflection', true, 'RandXTranslation', [-100 100],'RandYTranslation', [-100 100]);
%
pximds = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'DataAugmentation', augmenter);

%start Training
doTraining = false;
if doTraining    
    disp("Training.....")
    [net, info] = trainNetwork(pximds,lgraph,options);
   
    
    save("secondTransformations.mat", 'net', 'info', 'options');
    disp("Trained")
else
    data = load("secondTransformations.mat")
    net = data.net;
end

dataFolder2 = 'D:\SungRung\mnist_SEG(Noise)\images\0\resized';
goalDS = imageDatastore(dataFolder2,'IncludeSubfolders',true,'LabelSource','foldernames');
num = numel(imds.Labels);
%test network
OutputFolder = 'D:\SungRung\mnist_SEG(Noise)\images\0\smoothed'; 
for i = 1:num
%     I1 = readimage(goalDS, i);
%     subplot(1,5,1);
%     I1 = imgaussfilt(I1,10);
%     imshow(I1); 
% %     pixelLabelColorbar(cmap, classes)
%     title("wanted image" + " " + i)
    
    I2 = readimage(imds, i);
    I2 = cast(I2, 'double');
    subplot(1,5,1);
    imshow(I2); 
    
    C = semanticseg(I2, net); 
    CB= labeloverlay(I2, C, 'ColorMap', cmap); 
    subplot(1,5,2);
    imshow(CB);
    title("Labeled image"+ " " + i)
    
    CB = semanticseg(I2, net); 
    CB= labeloverlay(I2, C, 'ColorMap', cmap); 
    subplot(1,5,3);
    imshow(CB);
    title("Labeled image 1 "+ " " + i)
    
    CC = semanticseg(I2, net); 
    CC= labeloverlay(I2, CC, 'ColorMap', cmap); 
    subplot(1,5,4);
    imshow(CC);
    title("Labeled image 2 "+ " " + i)
    
    CD = semanticseg(I2, net); 
    CD= labeloverlay(I2, CD, 'ColorMap', cmap); 
    subplot(1,5,5);
    imshow(CD);
    title("Labeled image"+ " " + i)
%     pixelLabelColorbar(cmap, classes)

    pause;
end


%functions//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
function labelIDs = camvidPixelLabelIDs()

labelIDs = { 000 %black
             255 %white
    };
end

function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds)
% Partition CamVid data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
numTrain = round(0.05 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
numVal = round(0.01 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages,"ReadFcn",@increase);
imdsVal = imageDatastore(valImages,"ReadFcn",@increase);
imdsTest = imageDatastore(testImages,"ReadFcn",@increase);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = camvidPixelLabelIDs();

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs,'ReadFcn', @to2D);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs,'ReadFcn', @to2D);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs,'ReadFcn', @to2D);
test1 = readimage(imds, 1);
    size(test1)
    test2 = readimage(pxdsTrain, 1);
    size(test2)
end

function img = mnistRead(file)
    img = imread(file);
    img = rgb2gray(img);
    
%     img = repmat(img, [1 1 3]);
end

function img = increase(file)
    img = imread(file);
    
    img = repmat(img, [1 1 3]);
end

function img = to2D(file)
     img = imread(file);
     if (size(img, 3) == 3)
    img = rgb2gray(img);
     end

end

function cmap = camvidColorMap()
    cmap = [0 0 0
            255 255 255
           ];

    cmap = cmap ./ 255;
end

function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.

colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end







