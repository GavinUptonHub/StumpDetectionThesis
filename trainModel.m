%% Thesis Extract Features & Train Model

% stump_images = zeros(221,221,3,100);
% nonStump_images = zeros(221,221,3,100);
% 
% % Load images
% for i = 1:100
%     
%     % If want to show image, convert back to uint8 from double.
%     stump_images(:,:,:,i) = imread(sprintf('stumps/stump%s.png',string(i)));
%     nonStump_images(:,:,:,i) = imread(sprintf('nonstumps/nonstump%s.png',string(i)));
% end

%% Extract Features
% 
% stump_features = zeros(100,111*30);
% nonStump_features = zeros(100,111*30);
% data_features = zeros(200,111*30);
% % Stump = 1, Non Stump = 0
% data_labels = zeros(200,1);
% 
% row = 1;
% 
% for i = 1:100
%     stump = stump_images(:,:,:,i);
%     stump_downsampled = stump(1:2:end,1:2:end,1:3);
%     stump_quantiles = quantile(stump_downsampled,10,2);
%     stump_features(i,:) = reshape(stump_quantiles,111*30,1)';
%     %disp(stump_features(i,:));
%     
%     data_features(row,:) = stump_features(i,:);
%     data_labels(row) = 1;
%     row = row+1;
%     
%     nonStump = nonStump_images(:,:,:,i);
%     nonStump_downsampled = nonStump(1:2:end,1:2:end,1:3);
%     nonStump_quantiles = quantile(nonStump_downsampled,10,2);
%     nonStump_features(i,:) = reshape(nonStump_quantiles,111*30,1)';
%     
%     data_features(row,:) = nonStump_features(i,:);
%     data_labels(row) = 0;
%     row = row+1;
%     
% end
% 
% combined_features = [data_features, data_labels];
% shuffled_combined_features = combined_features(randperm(size(combined_features,1)),:);
% 
%  % Extract/separate the randomized matrices of features and labels
% new_data_features = shuffled_combined_features(:,1:end-1);
% new_data_labels = shuffled_combined_features(:,end);


%% Train Model

% We want 5 sets of indices for training/testing - indices_train will be
% 5x40 and indices_val will be 5x160
% K = 4;
% N = 200;
% 
% % Determine indices for training sets and validation sets.
% [indices_train, indices_val] = determineTrainingValidindices(K,N);
% classes = cell(1,K);
% accuracy = zeros(1,K);
% confmat_full = zeros(2,2,K);
% rec_nonStump = zeros(1,K);
% rec_stump = zeros(1,K);
% 
% for j = 1:K
%     
%     % Train model with svm
%     svm_model = fitcsvm(new_data_features(indices_train(j,:),:),new_data_labels(indices_train(j,:)));
%     classes{j} = predict(svm_model,new_data_features(indices_val(j,:),:));
%     actual_classes = new_data_labels(indices_val(j,:));
%     [confmat, acc, rec] = analyseModel(classes{j},actual_classes);
%     confmat_full(:,:,j) = confmat;
%     accuracy(j) = acc;
%     rec_nonStump(j) = rec(1);
%     rec_stump(j) = rec(2);
%     disp(accuracy(j));
%     
% end
% fprintf('Average accuracy is %4.2f%%.\n',mean(accuracy)*100);
% fprintf('Average Recall for Non Stumps is %4.2f%%.\n',mean(rec_nonStump)*100);
% fprintf('Average Recall for Stumps is %4.2f%%.\n',mean(rec_stump)*100);
% 
%% Join stump image with topo information
% for i = 1:122
%     if i < 101
          stump = imread('stumps600/Test/stump49.png');
%           allBlack = zeros(600,600,'uint8');
%           redChan = cat(3,stump(:,:,1),allBlack,allBlack);
%           greenChan = cat(3,allBlack,stump(:,:,2),allBlack);
%           blueChan = cat(3,allBlack,allBlack,stump(:,:,3));
%          
       
%     else
%         stump = imread(sprintf('stumps600/Train/stump%s.png',string(i)));
%     end
     topo = imread('topo_600/topo49.png');
     topo = mat2gray(topo);%,[double(min(min(topo))) double(max(max(topo)))]);

     
     newimage = (zeros(600,600,3));
     newimage(:,:,1:2) = mat2gray(stump(:,:,1:2));
     
     newimage(:,:,3) = topo;
    imshow(newimage);
%      figure(1);
%      subplot(1,4,1);
%      imshow(redChan);
%      title('R');
%      subplot(1,4,2);
%      imshow(greenChan);
%      title('G');
%      subplot(1,4,3);
%      imshow(blueChan);
%      title('B');
%      subplot(1,4,4);
%      imshow(stump);
%      title('RGB');
%      
%      figure(2);
%      subplot(1,4,1);
%      imshow(redChan);
%      title('R');
%      subplot(1,4,2);
%      imshow(greenChan);
%      title('G');
%      subplot(1,4,3);
%      imshow(topo);
%      title('Topography');
%      subplot(1,4,4);
%      imshow(newimage);
%      title('RGT');
     
     % %newimage = cat(3,stump(:,:,1),stump(:,:,2),topo*255);
% 
%imagesc(newimage);
     %imwrite(newimage,sprintf('stumps_topo_600/stump_topo8.png'));%,string(i)));
%     %colorbar;
% end
% load('stump600ROIs.mat');
% stump_topoROIs = stumpROIs;
% for j = 1:100
%     stump_topoROIs{j,1} = {sprintf('/Users/Gavin/Desktop/ThesisStuff/stumps_topo_600/stump_topo%s.png',string(j))};
%     disp(j);
% end

%% Train R-CNN Model

load('stump600ROIs.mat');
load('stump_topoROIs.mat');
% MAKE TRAINING IMAGES 600 X 600 CONTAINING ONE INSTANCE OF STUMP BUT ALSO
% ALLOWING SPACE FOR PROGRAM TO DETECT AREAS OF NO STUMP. ALSO INCREASE
% IMAGEINPUTLAYER TO [600 600 3] AND MAKE SURE THE TEST IMAGES ARE THE
% EXACT SAME SIZE
% F = Size of Convolution filter and D = number of filters
% F1 = 8;
% D1 = 10;
% F2 = 5;
% D2 = 10;
% conv1 = convolution2dLayer(F1,D1,'Padding',0,...
%                      'BiasLearnRateFactor',2,...
%                      'Stride',2,...
%                      'name','conv1');
% conv1.Weights = (single(randn([F1 F1 3 D1])*0.0001));
% conv1.Bias = (single(randn([1 1 D1])*0.00001+1));
% conv2 = convolution2dLayer(F2,D2,'Padding',0,...
%                      'BiasLearnRateFactor',2,...
%                      'Stride',2,...
%                      'name','conv2');
% % conv2.Weights = (single(randn([F2 F2 3 D2])*0.0001));
% % conv2.Bias = (single(randn([1 1 D2])*0.00001+1));
layers = [imageInputLayer([600 600 3])
    convolution2dLayer([8 8],20, 'Stride', 2)
    reluLayer()
    maxPooling2dLayer(5,'Stride',5)
    % Second CONV layer with associated reluLayer and Pooling layer!
    convolution2dLayer([5 5],20, 'Stride', 2)
    reluLayer()
    maxPooling2dLayer(3,'Stride',3)
    dropoutLayer
    fullyConnectedLayer(2)
    softmaxLayer()
    classificationLayer()];

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.3, ...
    'LearnRateDropPeriod',5, ...
    'L2Regularization',0.0001,...
    'InitialLearnRate', 1e-3, ...    
    'MaxEpochs', 50, ...
    'Verbose', true);
% 
% net = alexnet;
% layers_alex = net.Layers;
% % Reduce output size of final max pooling layer by increasing pool size to 5.
% % This changes the minimum size to 88-by-88.
% layers_alex(1) = imageInputLayer([600 600 3]);
% layers_alex(16) = maxPooling2dLayer(5,'Stride',5);  
% % reset fully connected layers because of the size change. 
% % Note: This may not be the ideal set of layers and might require some experimentation
% % to figure out the best number of layers after making this change to the max pooling
% % layer. 
% layers_alex(17) = fullyConnectedLayer(4096);
% layers_alex(20) = fullyConnectedLayer(4096);
% layers_alex(23) = fullyConnectedLayer(2);
% layers_alex(end) = classificationLayer();

%fast_rcnn = trainFastRCNNObjectDetector(stumpROIs,layers,options,'NegativeOverlapRange', [0.1 0.3], ...
   % 'PositiveOverlapRange', [0.6 1]);%, ...
    %'SmallestImageDimension', 600);

% network = rcnn.Network;
% layers = network.Layers;
% 
rcnnFinal = trainFastRCNNObjectDetector(stump_topoROIs, layers, options, 'NegativeOverlapRange',[0 0.3],'PositiveOverlapRange',[0.6 1]);
%% Evaluate precision/accuracy 
%evaulateDetectionPrecision
load('test_stump_topoROIs.mat');

numTestImages = 21;
Boxes = cell(21,1);
Scores = cell(21,1);
results = table(Boxes,Scores);
count = 1;
for i = 101:122
    disp(i);
    if i == 111
        ;
    else
        img = imread(sprintf('stumps_topo_600/stump_topo%s.png',string(i)));
        [bbox,score,label] = detect(rcnnFinal,img);
        results.Boxes{count} = bbox;
        results.Scores{count} = score;
        count = count+1;
    end
    
end
[ap,recall,precision] = evaluateDetectionPrecision(results,test_stump_topoROIs(:,2));
disp(ap);
%%
% For fastrcnn200Epoch_2, images with no detections: 101, 102, 107, 108,
% 113
% For fastrcnn_alexnet20Epoch, images w no detections: 105, 107, 113
%alexnet50Epoch USELESS - 101, 103, 105, 107, 108, 109, 113, 116, 117, 118, 120 
img = imread('stumps_topo_600/stump_topo101.png');
%img = imread('stump1.png');

[bbox, score, label] = detect(rcnnFinal, img);

% Find top three confidence level detections. Change the 3's in the if loop
% statement and inside loop to increase number of detections displayed
[scores_sorted, indices_sorted] = sort(score,'descend');
if length(score) > 3
    
    indices = indices_sorted(1:3);
    scores = scores_sorted(1:3);
else
    indices = indices_sorted;
    scores = scores_sorted;
end

% Find detections with scores above a desired threshold (e.g. 0.3)
%indices = find(score>0.3);
%scores = score(indices);
bbox = bbox(indices, :);
[rows, cols] = size(bbox);
% Increase this if statement value to like 50 if you don't want bbox's to
% be merged
if rows < 20
    if rows > 0
        annotation = cell(rows,1);
        for ii = 1:rows
            annotation{ii} = ['Confidence = ' num2str(100*scores(ii)) '%'];
        end
    else
        annotation = {'None'};
    end
    detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation,'LineWidth',5,'TextBoxOpacity',0.5,'FontSize',14);
else
    
    % Attempt to merge top 2 confidence bounding boxes
    [expanded_x, idx_x] = min([bbox(1,1),bbox(2,1)]);
    [expanded_y, idx_y] = min([bbox(1,2),bbox(2,2)]);
    expanded_bbox = [expanded_x, expanded_y, (max([bbox(1,1),bbox(2,1)])-expanded_x)+bbox(not(idx_x-1)+1,3),(max([bbox(1,2),bbox(2,2)])-expanded_y)+bbox(not(idx_y-1)+1,4)];
    expanded_annotation = {['Confidence = ' num2str(100*scores(1)*scores(2)) '%']};
    detectedImg = insertObjectAnnotation(img, 'rectangle', expanded_bbox,expanded_annotation,'LineWidth',5,'TextBoxOpacity',0.5,'FontSize',14);
    %annotation = {(sprintf('%s: (Confidence = %f)', label,scores))};   
end



figure
imshow(detectedImg);
%% Use Model

input = imread('mosaics/mosaic_4_2.tif');
%imshow(input);
input = double(input);
input_downsampled = input(1:2:end,1:2:end,1:3);
[rows, cols, chan] = size(input_downsampled);
patch_separation = 30;
labels = zeros(round((rows-56)/patch_separation),round((cols-56)/patch_separation));
scores = zeros(round((rows-56)/patch_separation),round((cols-56)/patch_separation));
output_col = 1;
output_row = 1;

% r = rows, c = cols
for r = 56:patch_separation:rows-55
    output_col = 1;
    for c = 56:patch_separation:cols-55
        
        input_patch = input_downsampled(r-55:r+55,c-55:c+55,:);
        %imshow(uint8(input_patch));
        input_patch_quantiles = quantile(input_patch,10,2);
        input_patch_features = reshape(input_patch_quantiles,111*30,1)';
        [label,score] = predict(svm_model,input_patch_features);
        labels(output_row,output_col) = label;
        scores(output_row, output_col) = max(score);
        output_col = output_col+1;
        %disp(output_col);
    end
    output_row=output_row+1;
    disp(r);
    %disp(output_row);
end
%%
imshow(uint8(input_downsampled));
title('Mosaic 4-2: Confidence > 0.2');
hold on;
for k = 1:length(find(labels))
    [detection_rows,detection_cols] = find(labels);
    detection_pos = [detection_rows, detection_cols];
    if scores(detection_pos(k,1),detection_pos(k,2))>0.2
       plot(56+(detection_pos(k,2)-1)*patch_separation,56+(detection_pos(k,1)-1)*patch_separation,'rx');
    end
%     for i = 1:109
%         for j = 1:length(locs_full{i})
%             plot(56+(i-1)*patch_separation,56+(locs_full{i}(j)-1)*patch_separation,'rx');
%         end
%     end
end

%%
scores_new = zeros(111,109);
for i = 1:length(detection_pos)
    scores_new(detection_pos(i,1),detection_pos(i,2)) = scores(detection_pos(i,1),detection_pos(i,2));
end
disp('Done');
