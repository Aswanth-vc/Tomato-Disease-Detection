clear all
clc

imdsTrain = imageDatastore('final/train','IncludeSubfolders',true,'LabelSource','foldernames'); 
imdsValidation=imageDatastore('final/valid','IncludeSubfolders',true,'LabelSource','foldernames');

net_squeeze = squeezenet;
analyzeNetwork(net_squeeze)
inputSize = net_squeeze.Layers(1).InputSize
lgraph = layerGraph(net_squeeze);
numClasses = 10
newConvLayer =  convolution2dLayer([1, 1],numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,"Name",'new_conv');
lgraph = replaceLayer(lgraph,'conv10',newConvLayer);
newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassificatonLayer);
%augmentation
imageAugmenter =imageDataAugmenter('RandRotation',[-5 5],'RandXReflection',1,'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05])
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, 'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'DataAugmentation',imageAugmenter);
% training
options = trainingOptions('sgdm','MaxEpochs',6,'MiniBatchSize',32,'Shuffle','every-epoch', 'InitialLearnRate',1e-4, 'Verbose',false, 'Plots','training-progress');
net_squeeze = trainNetwork(augimdsTrain,layers,options);
[YPred,scores] = classify(net_squeeze,augimdsTrain)
Ytrain = imdsTrain.Labels;
accuracy= mean(YPred == Ytrain)
%testing
[YPred2,scores] = classify(net_squeeze,augimdsValidation)
YValidation = imdsValidation.Labels;
accuracy2= mean(YPred2 == YValidation)
