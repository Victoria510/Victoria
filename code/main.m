location = ''; 
imds = imageDatastore(location,'IncludeSubfolders',1,...
    'LabelSource','foldernames');
tbl = countEachLabel(imds);

lgraph = layerGraph();

tempLayers = imageInputLayer([   ],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([11 11],96,"Name","conv1'","BiasLearnRateFactor",2,"Stride",[4 4])
    reluLayer("Name","relu1'")
%     crossChannelNormalizationLayer(5,"Name","norm1'")
    batchNormalizationLayer("Name","batchnorm1'")
    maxPooling2dLayer([3 3],"Name","pool1'","Stride",[2 2])
    convolution2dLayer([5 5],128,"Name","conv2'","Padding","same")
    reluLayer("Name","relu2'")
    batchNormalizationLayer("Name","batchnorm2'")
    maxPooling2dLayer([3 3],"Name","pool2'","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4])
    reluLayer("Name","relu1")
%     crossChannelNormalizationLayer(5,"Name","norm1")
    batchNormalizationLayer("Name","batchnorm1")
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    convolution2dLayer([5 5],128,"Name","conv2","Padding","same")
    reluLayer("Name","relu2")
    batchNormalizationLayer("Name","batchnorm2")
    maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    convolution2dLayer([3 3],512,"Name","conv3","Padding",[1 1 1 1])
    reluLayer("Name","relu3")
    convolution2dLayer([3 3],512,"Name","conv4","Padding",[1 1 1 1])
    reluLayer("Name","relu4")
    convolution2dLayer([3 3],512,"Name","conv5","Padding",[1 1 1 1])
    reluLayer("Name","relu5")
    fullyConnectedLayer(4096,"Name","fc1")
    reluLayer("Name","relu6")
    fullyConnectedLayer(64,"Name","fc2")
    reluLayer("Name","relu7")
    fullyConnectedLayer(5,"Name","fc3")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"imageinput","conv1'");
lgraph = connectLayers(lgraph,"imageinput","conv1");
lgraph = connectLayers(lgraph,"pool2","addition/in1");
lgraph = connectLayers(lgraph,"pool2'","addition/in2");


imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
[trainingDS, testDS] = splitEachLabel(imds, 0.7,'randomize'); %splitEachLabel是拆分数据集，根据0.7划分
trainingDS.Labels = categorical(trainingDS.Labels);
trainingDS.ReadFcn = @readFunctionTrain;  

testDS.Labels = categorical(testDS.Labels);
testDS.ReadFcn = @readFunctionTrain;

miniBatchSize =128; 
numImages = numel(trainingDS.Files); 

maxEpochs = ; 
lr = ;  
opts = trainingOptions('sgdm', ...  
    'LearnRateSchedule', 'none',... 
    'InitialLearnRate', lr,... 
    'MaxEpochs', maxEpochs, ... 
    'MiniBatchSize', 128,'Plots','training-progress'); 
[net ,info]= trainNetwork(trainingDS, lgraph, opts);  
plot(lgraph);

[labels,err_test] = classify(net, testDS, 'MiniBatchSize', 64); 
confMat = confusionmat(testDS.Labels, labels); 
figure;
plotconfusion(testDS.Labels,labels); 
confMat = bsxfun(@rdivide,confMat,sum(confMat,2)); 

