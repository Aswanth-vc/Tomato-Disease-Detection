clc
clear all
imdsTrain= imageDatastore('final/train', 'IncludeSubfolders',true,'LabelSource','foldernames');
imdsValidation=imageDatastore('final/valid','IncludeSubfolders',true,'LabelSource','foldernames');
net = vgg16
analyzeNetwork(net)
%modify last layer
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
% augmentation and resize
 augmenter = imageDataAugmenter('RandRotation',[-5 5],'RandXReflection',1,'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05])
augimdsTrain = augmentedImageDatastore([224,224],imdsTrain,'DataAugmentation',augmenter)
augimdsValid= augmentedImageDatastore([224,224],imdsValidation,'DataAugmentation',augmenter)
% training
options = trainingOptions('sgdm','MiniBatchSize',32,'MaxEpochs',6,'InitialLearnRate',1e-4,'Shuffle','every-epoch','Verbose',false,'Plots','training-progress');
net_vg=trainNetwork(augimdsTrain,lgraph,options);
Ytrain = imdsTrain.Labels;
[YPred1,probs]=classify(net_vg,augimdsTrain);
net_vgg=trainNetwork(augimdsTrain,lgraph,options);
Ytrain = imdsTrain.Labels;
[YPred1,probs]=classify(net_vgg,augimdsTrain);
%training accuracy
accuracy= mean(YPred1== Ytrain)
%testing
YTest = imdsValidation.Labels;
[YPred2,probs]=classify(net_vgg,augimdsValid);
%testing accuracy
accuracy=mean(YPred2== YTest)
