clear all
clc
imds = imageDatastore('final\train','IncludeSubfolders',true,'LabelSource','foldernames');
imdsTrain = imageDatastore('final\train','IncludeSubfolders',true,'LabelSource','foldernames');
imdsValidation= imageDatastore('final\valid','IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest= imageDatastore('final\test','IncludeSubfolders',true,'LabelSource','foldernames');
net = alexnet;
analyzeNetwork(net)
%three layers must be fine-tuned for the new classification problem
inputSize = net.Layers(1).InputSize
InputSizelayersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))
layersTransfer = net.Layers(1:end-3);
layers = [layersTransfer,fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20),softmaxLayer,classificationLayer];
% data augmentation
imageAugmenter =imageDataAugmenter('RandRotation',[-5 5],'RandXReflection',1,'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05])
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'DataAugmentation',imageAugmenter);
%training
options = trainingOptions('sgdm','MiniBatchSize',32,'MaxEpochs',6,'InitialLearnRate',1e-4,'Shuffle','every-epoch','Verbose',false,'Plots','training-progress');
netTransfer_a = trainNetwork(augimdsTrain,layers,options);
[YPred,scores] = classify(netTransfer,augimdsTrain)
Ytrain = imdsTrain.Labels;
accuracy= mean(YPred == Ytrain)
%testing
[YPred2,scores] = classify(netTransfer,augimdsValidation)
YValidation = imdsValidation.Labels;
accuracy2= mean(YPred2 == YValidation)


